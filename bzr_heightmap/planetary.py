from __future__ import annotations

import math
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
from scipy import ndimage

from .builder import TerrainBuilder
from .hg2 import HG2Map
from .noise import fbm, meander_path, ridged_fbm, smoothstep01, vary_widths
from .settings import GeneratorSettings


def _edge_point(builder: TerrainBuilder, side: str, t: float) -> Tuple[float, float]:
    if side == "left":
        return 0.0, t * (builder.h - 1)
    if side == "right":
        return builder.w - 1.0, t * (builder.h - 1)
    if side == "top":
        return t * (builder.w - 1), 0.0
    return t * (builder.w - 1), builder.h - 1.0


def _masked_fbm(
    b: TerrainBuilder,
    amplitude: float,
    feature_px: float,
    mask_feature_px: float,
    coverage: float,
    *,
    ridged: bool = False,
    mask_softness: float = 0.45,
) -> None:
    detail = ridged_fbm(b.a.shape, feature_px, b.rng, 4) if ridged else fbm(b.a.shape, feature_px, b.rng, octaves=4)
    mask_field = fbm(b.a.shape, mask_feature_px, b.rng, octaves=3, persistence=.52)
    threshold = float(np.quantile(mask_field, np.clip(1.0 - coverage, .05, .95)))
    spread = max(float(np.std(mask_field)) * max(mask_softness, .08), 1e-4)
    mask = smoothstep01(np.clip((mask_field - threshold + spread) / (spread * 2.0), 0.0, 1.0))
    b.a += detail.astype(np.float32) * mask.astype(np.float32) * float(amplitude)


def _ramp_path(
    b: TerrainBuilder,
    points: Sequence[Tuple[float, float]],
    start_height: float,
    end_height: float,
    half_width: float,
    feather: float,
    endpoint_taper: float = .72,
) -> None:
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) < 2:
        return
    seg = np.asarray([math.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1]) for i in range(len(pts)-1)], dtype=np.float32)
    total = float(np.sum(seg))
    if total < 1e-4:
        return
    cumulative = np.concatenate(([0.0], np.cumsum(seg[:-1]))).astype(np.float32)
    margin = float(half_width + feather + 2)
    minx=max(0,int(math.floor(min(x for x,_ in pts)-margin))); maxx=min(b.w-1,int(math.ceil(max(x for x,_ in pts)+margin)))
    miny=max(0,int(math.floor(min(y for _,y in pts)-margin))); maxy=min(b.h-1,int(math.ceil(max(y for _,y in pts)+margin)))
    if minx>=maxx or miny>=maxy:
        return
    yy,xx=np.mgrid[miny:maxy+1,minx:maxx+1].astype(np.float32)
    best=np.full(xx.shape,np.inf,dtype=np.float32); progress=np.zeros(xx.shape,dtype=np.float32)
    for i in range(len(pts)-1):
        sx,sy=pts[i]; ex,ey=pts[i+1]; vx,vy=ex-sx,ey-sy; l2=max(vx*vx+vy*vy,1e-6)
        lt=np.clip(((xx-sx)*vx+(yy-sy)*vy)/l2,0,1); nx=sx+lt*vx; ny=sy+lt*vy; d2=(xx-nx)**2+(yy-ny)**2
        better=d2<best; best[better]=d2[better]; pr=(cumulative[i]+lt*seg[i])/total; progress[better]=pr[better]
    lateral=np.sqrt(best); t=np.clip(progress,0,1)
    taper=float(endpoint_taper)+(1.0-float(endpoint_taper))*np.power(np.clip(np.sin(np.pi*t),0,1),.72)
    local_width=np.maximum(float(half_width)*taper,1.0)
    weight=1.0-smoothstep01((lateral-local_width)/max(float(feather),1.0)); weight[lateral>=local_width+feather]=0.0
    target=float(start_height)+(float(end_height)-float(start_height))*t
    region=b.a[miny:maxy+1,minx:maxx+1]; region[:]=region*(1-weight)+target*weight
    b.protected[miny:maxy+1,minx:maxx+1] |= lateral <= local_width*.34


def _path_side_ramps(
    b: TerrainBuilder,
    points: Sequence[Tuple[float,float]],
    count: int,
    inner_offset: float,
    outer_offset: float,
    half_width: float,
    feather: float,
) -> None:
    if len(points)<5:
        return
    candidates=np.linspace(.12,.88,max(count*4,count),dtype=np.float32); b.rng.shuffle(candidates); placed=0
    for t in candidates:
        if placed>=count: break
        idx=int(np.clip(round(float(t)*(len(points)-1)),1,len(points)-2)); px,py=points[idx]; ax,ay=points[idx-1]; bx,by=points[idx+1]
        tx,ty=bx-ax,by-ay; mag=math.hypot(tx,ty)
        if mag<1e-5: continue
        tx,ty=tx/mag,ty/mag; nx,ny=-ty,tx; side=-1.0 if placed%2 else 1.0
        span=max(outer_offset-inner_offset,1.0); tangent=span*float(b.rng.uniform(1.45,2.25)); sign=-1.0 if b.rng.random()<.5 else 1.0
        sx=px+nx*inner_offset*side-tx*tangent*.48*sign; sy=py+ny*inner_offset*side-ty*tangent*.48*sign
        ex=px+nx*outer_offset*side+tx*tangent*.52*sign; ey=py+ny*outer_offset*side+ty*tangent*.52*sign
        margin=half_width+feather+3
        if min(sx,ex)<margin or min(sy,ey)<margin or max(sx,ex)>=b.w-margin or max(sy,ey)>=b.h-margin: continue
        sh=float(b.a[int(round(sy)),int(round(sx))]); eh=float(b.a[int(round(ey)),int(round(ex))])
        if abs(eh-sh)<130: continue
        p1=(sx+tx*tangent*.34*sign+nx*span*.12*side, sy+ty*tangent*.34*sign+ny*span*.12*side)
        p2=(sx+tx*tangent*.72*sign+nx*span*.62*side, sy+ty*tangent*.72*sign+ny*span*.62*side)
        _ramp_path(b,[(sx,sy),p1,p2,(ex,ey)],sh,eh,half_width,feather,.68); placed+=1


def _repair_connectivity(
    b: TerrainBuilder,
    max_slope_deg: float = 38.0,
    target_pct: float = .92,
    max_repairs: int = 6,
    half_width: float = 7.0,
    feather: float = 12.0,
) -> None:
    """Terrain-quality repair, not an assertion about BZR's exact AI slope limit."""
    for _ in range(max_repairs):
        a=b.a.astype(np.float32); gy,gx=np.gradient(a*.1,5.0,5.0); slope=np.degrees(np.arctan(np.hypot(gx,gy))); passable=slope<=max_slope_deg
        labels,n=ndimage.label(passable,structure=np.ones((3,3),dtype=np.uint8)); counts=np.bincount(labels.ravel())[1:]
        total=int(np.count_nonzero(passable))
        if n<=1 or total<=0: return
        main=int(np.argmax(counts))+1
        if float(counts[main-1])/total>=target_pct: return
        min_size=max(32,int(a.size*.008)); others=[i+1 for i,c in enumerate(counts) if i+1!=main and c>=min_size]
        if not others: return
        other=max(others,key=lambda lab:counts[lab-1]); ma=labels==main; ob=labels==other
        mba=ma & ~ndimage.binary_erosion(ma); obb=ob & ~ndimage.binary_erosion(ob)
        my,mx=np.nonzero(mba); oy,ox=np.nonzero(obb)
        if not len(mx) or not len(ox): return
        step_m=max(1,len(mx)//800); step_o=max(1,len(ox)//800); mp=np.column_stack([mx[::step_m],my[::step_m]]).astype(np.float32); op=np.column_stack([ox[::step_o],oy[::step_o]]).astype(np.float32)
        best=None
        for i in range(0,len(op),160):
            chunk=op[i:i+160]; d=((chunk[:,None,:]-mp[None,:,:])**2).sum(axis=2); pos=np.unravel_index(int(np.argmin(d)),d.shape); val=float(d[pos])
            if best is None or val<best[0]: best=(val,chunk[pos[0]],mp[pos[1]])
        if best is None: return
        p0=tuple(float(v) for v in best[1]); p3=tuple(float(v) for v in best[2]); dx=p3[0]-p0[0]; dy=p3[1]-p0[1]; mag=max(math.hypot(dx,dy),1.0); nx,ny=-dy/mag,dx/mag
        bend=min(mag*.28,min(b.h,b.w)*.075); sign=-1.0 if b.rng.random()<.5 else 1.0
        p1=(p0[0]+dx*.34+nx*bend*sign,p0[1]+dy*.34+ny*bend*sign); p2=(p0[0]+dx*.70+nx*bend*.65*sign,p0[1]+dy*.70+ny*bend*.65*sign)
        sh=float(b.a[int(np.clip(round(p0[1]),0,b.h-1)),int(np.clip(round(p0[0]),0,b.w-1))]); eh=float(b.a[int(np.clip(round(p3[1]),0,b.h-1)),int(np.clip(round(p3[0]),0,b.w-1))])
        _ramp_path(b,[p0,p1,p2,p3],sh,eh,half_width,feather,.72)


def pluto_basin(s: GeneratorSettings) -> HG2Map:
    b=TerrainBuilder(s).set_level(1040); m=min(b.h,b.w); yy,xx=np.mgrid[0:b.h,0:b.w].astype(np.float32)
    cx=b.w*float(b.rng.uniform(.46,.54)); cy=b.h*float(b.rng.uniform(.47,.55)); rx=b.w*float(b.rng.uniform(.31,.38)); ry=b.h*float(b.rng.uniform(.28,.35))
    ell=np.sqrt(((xx-cx)/max(rx,1))**2+((yy-cy)/max(ry,1))**2); b.a-=(np.exp(-.5*(ell/.72)**4)*350).astype(np.float32); b.a+=(np.exp(-.5*((ell-1)/.21)**2)*150).astype(np.float32)
    b.add_fbm(68,m*.54,ridged=False,octaves=3); rough=ridged_fbm(b.a.shape,m*.145,b.rng,4); outer=np.clip((ell-.76)/.52,0,1); outer=outer*outer*(3-2*outer); patch=np.clip((fbm(b.a.shape,m*.34,b.rng,octaves=3,persistence=.52)+.18)/.80,0,1); b.a+=(rough*outer*patch*(170+55*s.feature_density)).astype(np.float32)
    for side,t in [("left",.58),("right",.43)]:
        st=_edge_point(b,side,float(np.clip(t+b.rng.uniform(-.12,.12),.12,.88))); en=(cx+float(b.rng.uniform(-rx*.24,rx*.24)),cy+float(b.rng.uniform(-ry*.20,ry*.20))); path=meander_path(st,en,10,m*.070*s.naturalization,b.rng); sh=float(b.a[int(np.clip(round(st[1]),0,b.h-1)),int(np.clip(round(st[0]),0,b.w-1))]); eh=float(b.a[int(np.clip(round(en[1]),0,b.h-1)),int(np.clip(round(en[0]),0,b.w-1))]); _ramp_path(b,path,sh,eh,m*.020,m*.050,.78)
    for _ in range(3+int(4*s.feature_density)):
        rad=float(b.rng.uniform(m*.018,m*.045)); ang=float(b.rng.uniform(0,math.tau)); rr=float(b.rng.uniform(.78,1.18)); px=cx+math.cos(ang)*rx*rr; py=cy+math.sin(ang)*ry*rr
        if rad<px<b.w-rad and rad<py<b.h-rad: b.crater(px,py,rad,rad*1.15,rad*.34,float(b.rng.uniform(.82,1.22)))
    b.add_detail(3*s.detail,m*.055); _repair_connectivity(b,38,.92,4,m*.008,m*.014); return b.finalize(center_height=1350,preserve_flats=True)


def venus_shield(s: GeneratorSettings) -> HG2Map:
    b=TerrainBuilder(s).set_level(720); m=min(b.h,b.w); yy,xx=np.mgrid[0:b.h,0:b.w].astype(np.float32); cx=b.w*float(b.rng.uniform(.45,.55)); cy=b.h*float(b.rng.uniform(.44,.56)); dx=xx-cx; dy=yy-cy; rx=m*float(b.rng.uniform(.28,.34)); ry=m*float(b.rng.uniform(.24,.31)); r=np.sqrt((dx/max(rx,1))**2+(dy/max(ry,1))**2); shield=np.clip(1-r,0,1); shield=shield*shield*(3-2*shield); b.a+=(shield*760+np.exp(-.5*((r-.90)/.36)**2)*105).astype(np.float32); b.crater(cx,cy,m*.060,300,125,float(b.rng.uniform(.90,1.12))); b.crater(cx+m*.018,cy-m*.010,m*.026,110,40,1.05); b.add_fbm(70,m*.44,ridged=False,octaves=3); phase=float(b.rng.uniform(0,math.tau))
    arms=5+int(3*s.feature_density)
    for i in range(arms):
        ang=phase+i*math.tau/arms+float(b.rng.uniform(-.24,.24)); inner=float(b.rng.uniform(.13,.22)); outer=inner+float(b.rng.uniform(.09,.18)); path=meander_path((cx+math.cos(ang)*m*inner,cy+math.sin(ang)*m*inner),(cx+math.cos(ang)*m*outer,cy+math.sin(ang)*m*outer),7,m*.028*s.naturalization,b.rng); b.add_ridge_path(path,float(b.rng.uniform(32,68)),m*.004,m*.022)
    for ang in [phase+float(b.rng.uniform(.4,1.2)),phase+float(b.rng.uniform(2.5,3.5)),phase+float(b.rng.uniform(4.5,5.4))]:
        p0=(cx+math.cos(ang)*m*.12,cy+math.sin(ang)*m*.11); p1=(cx+math.cos(ang)*m*.40,cy+math.sin(ang)*m*.38); b.carve_path(meander_path(p0,p1,10,m*.040*s.naturalization,b.rng),float(b.rng.uniform(38,65)),m*.015,m*.055,5)
    b.add_random_craters(3+int(4*s.feature_density),(m*.010,m*.026),1.0,.28); b.add_detail(4*s.detail,m*.050); _repair_connectivity(b,38,.91,4,m*.008,m*.014); return b.finalize(center_height=1420,preserve_flats=True)


def lunar_catena(s: GeneratorSettings) -> HG2Map:
    b=TerrainBuilder(s).set_level(590); m=min(b.h,b.w); b.add_fbm(72,m*.48,ridged=False,octaves=3); b.add_random_craters(3+int(4*s.feature_density),(m*.060,m*.110),.95,.38); b.add_random_craters(7+int(10*s.feature_density),(m*.022,m*.052),1.05,.40); b.add_random_craters(13+int(18*s.feature_density),(m*.007,m*.020),1.15,.34)
    for ci in range(1+int(s.feature_density>.66)):
        st=_edge_point(b,'left',float(b.rng.uniform(.28,.68))) if ci==0 else _edge_point(b,'top',float(b.rng.uniform(.22,.70))); en=_edge_point(b,'right',float(b.rng.uniform(.30,.72))) if ci==0 else _edge_point(b,'bottom',float(b.rng.uniform(.26,.76))); path=meander_path(st,en,10,m*.070*s.naturalization,b.rng)
        for px,py in path[1:-1]:
            if b.rng.random()<.22: continue
            rad=m*float(b.rng.uniform(.010,.022)); px+=float(b.rng.uniform(-m*.007,m*.007)); py+=float(b.rng.uniform(-m*.007,m*.007)); b.crater(px,py,rad,rad*float(b.rng.uniform(.95,1.45)),rad*float(b.rng.uniform(.26,.44)),float(b.rng.uniform(.84,1.20)))
    yy,xx=np.mgrid[0:b.h,0:b.w].astype(np.float32)
    for _ in range(2):
        cx=float(b.rng.uniform(m*.18,b.w-m*.18)); cy=float(b.rng.uniform(m*.18,b.h-m*.18)); rad=m*float(b.rng.uniform(.12,.20)); b.a+=(np.exp(-.5*(np.hypot(xx-cx,yy-cy)/max(rad,1)/.85)**2)*float(b.rng.uniform(70,125))).astype(np.float32)
    b.add_detail(3*s.detail,m*.055); _repair_connectivity(b,38,.93,3,m*.008,m*.014); return b.finalize(center_height=1080,preserve_flats=False)


def mars_rift(s: GeneratorSettings) -> HG2Map:
    b=TerrainBuilder(s).set_level(2320); m=min(b.h,b.w); b.add_fbm(135,m*.42,ridged=False,octaves=3); trunk=meander_path(_edge_point(b,'left',float(b.rng.uniform(.18,.34))),_edge_point(b,'right',float(b.rng.uniform(.66,.82))),18,m*.13*(.45+.55*s.naturalization),b.rng); widths=vary_widths(len(trunk),m*.038,.40*s.naturalization,b.rng,cycles=6); b.carve_variable_corridor_level(trunk,720,widths,m*.095,110,m*.008*s.naturalization)
    for i in range(3+int(3*s.feature_density)):
        branch=meander_path(_edge_point(b,'top' if i%2==0 else 'bottom',float(b.rng.uniform(.12,.88))),trunk[int(b.rng.integers(3,len(trunk)-3))],10,m*.075*s.naturalization,b.rng); widths2=vary_widths(len(branch),m*float(b.rng.uniform(.018,.026)),.34*s.naturalization,b.rng); b.carve_variable_corridor_level(branch,760,widths2,m*.070,62,m*.005*s.naturalization); _path_side_ramps(b,branch,1,m*.020,m*.115,m*.0075,m*.014)
    _path_side_ramps(b,trunk,5+int(3*s.feature_density),m*.025,m*.135,m*.0085,m*.015); _masked_fbm(b,270,m*.095,m*.27,.38,ridged=True,mask_softness=.42); b.add_random_craters(3+int(5*s.feature_density),(m*.012,m*.032),1.0,.30); b.add_detail(7*s.detail,m*.043); _repair_connectivity(b,36,.94,8,m*.009,m*.016); return b.finalize(center_height=2480,preserve_flats=True)


def callisto_craterlands(s: GeneratorSettings) -> HG2Map:
    b=TerrainBuilder(s).set_level(720); m=min(b.h,b.w); b.add_fbm(105,m*.48,ridged=False,octaves=3)
    for count,rrange,depth,rim in [(2+int(3*s.feature_density),(m*.075,m*.135),.68,.30),(8+int(10*s.feature_density),(m*.028,m*.070),.82,.32),(18+int(22*s.feature_density),(m*.009,m*.030),.95,.30)]: b.add_random_craters(count,rrange,depth,rim)
    yy,xx=np.mgrid[0:b.h,0:b.w].astype(np.float32)
    for _ in range(3):
        cx=float(b.rng.uniform(m*.14,b.w-m*.14)); cy=float(b.rng.uniform(m*.14,b.h-m*.14)); rad=m*float(b.rng.uniform(.10,.18)); rr=np.hypot(xx-cx,yy-cy)/max(rad,1); b.a+=(np.exp(-.5*((rr-1)/.22)**2)*float(b.rng.uniform(55,95))).astype(np.float32)
    b.smooth(.65,.32); b.add_detail(3*s.detail,m*.055); _repair_connectivity(b,38,.94,3,m*.008,m*.014); return b.finalize(center_height=1180,preserve_flats=False)


def titan_basin_network(s: GeneratorSettings) -> HG2Map:
    b=TerrainBuilder(s).set_level(850); m=min(b.h,b.w); yy,xx=np.mgrid[0:b.h,0:b.w].astype(np.float32); b.add_fbm(205,m*.50,ridged=False,octaves=3); basins=[]
    for fx,fy,rx,ry in [(.30,.34,.23,.18),(.67,.40,.26,.21),(.50,.72,.28,.19)]:
        cx=b.w*(fx+float(b.rng.uniform(-.04,.04))); cy=b.h*(fy+float(b.rng.uniform(-.04,.04))); rr=np.sqrt(((xx-cx)/max(b.w*rx,1))**2+((yy-cy)/max(b.h*ry,1))**2); b.a-=(np.exp(-.5*(rr/.78)**4)*float(b.rng.uniform(170,265))).astype(np.float32); basins.append((cx,cy))
    _masked_fbm(b,235,m*.20,m*.40,.34,ridged=False,mask_softness=.66); anchors=[_edge_point(b,'left',float(b.rng.uniform(.35,.70))),*basins,_edge_point(b,'right',float(b.rng.uniform(.30,.68)))]
    for ia,ib in [(0,1),(1,2),(2,3),(3,4),(1,3)]: b.carve_path(meander_path(anchors[ia],anchors[ib],13,m*.095*s.naturalization,b.rng),float(b.rng.uniform(38,68)),m*.030,m*.095,float(b.rng.uniform(0,8)))
    for _ in range(6+int(5*s.feature_density)):
        cx=float(b.rng.uniform(m*.10,b.w-m*.10)); cy=float(b.rng.uniform(m*.10,b.h-m*.10)); rad=m*float(b.rng.uniform(.028,.060)); b.a+=(np.exp(-.5*(np.hypot(xx-cx,yy-cy)/max(rad,1)/.90)**2)*float(b.rng.uniform(75,165))).astype(np.float32)
    b.add_random_craters(2+int(4*s.feature_density),(m*.015,m*.035),.72,.22); b.add_detail(4*s.detail,m*.055); _repair_connectivity(b,38,.95,2,m*.008,m*.014); return b.finalize(center_height=1320,preserve_flats=False)


def europa_fracture_plains(s: GeneratorSettings) -> HG2Map:
    b=TerrainBuilder(s).set_level(720); m=min(b.h,b.w); b.add_fbm(72,m*.52,ridged=False,octaves=3)
    for i in range(3+int(s.feature_density>.65)):
        st=_edge_point(b,'left',float(b.rng.uniform(.08,.92))) if i%2==0 else _edge_point(b,'top',float(b.rng.uniform(.08,.92))); en=_edge_point(b,'right',float(b.rng.uniform(.08,.92))) if i%2==0 else _edge_point(b,'bottom',float(b.rng.uniform(.08,.92))); path=meander_path(st,en,13,m*.095*s.naturalization,b.rng)
        if i%2:
            b.add_ridge_path(path,float(b.rng.uniform(28,52)),m*.0045,m*.026); b.carve_path(path,float(b.rng.uniform(12,26)),m*.003,m*.015,0)
        else:
            b.carve_path(path,float(b.rng.uniform(30,55)),m*.006,m*.030,float(b.rng.uniform(10,22)))
    for _ in range(2+int(2*s.feature_density)):
        sx=float(b.rng.uniform(m*.08,b.w-m*.08)); sy=float(b.rng.uniform(m*.08,b.h-m*.08)); ang=float(b.rng.uniform(0,math.tau)); length=float(b.rng.uniform(m*.16,m*.34)); ex=sx+math.cos(ang)*length; ey=sy+math.sin(ang)*length
        if not (m*.04<ex<b.w-m*.04 and m*.04<ey<b.h-m*.04): continue
        path=meander_path((sx,sy),(ex,ey),8,m*.055*s.naturalization,b.rng)
        if b.rng.random()<.5:
            b.add_ridge_path(path,float(b.rng.uniform(22,45)),m*.004,m*.022)
        else:
            b.carve_path(path,float(b.rng.uniform(22,42)),m*.005,m*.025,8)
    for _ in range(3):
        cx=float(b.rng.uniform(m*.16,b.w-m*.16)); cy=float(b.rng.uniform(m*.16,b.h-m*.16)); b.flatten_pad(cx,cy,m*float(b.rng.uniform(.055,.095)),m*float(b.rng.uniform(.040,.075)),float(b.a[int(cy),int(cx)]+b.rng.uniform(-25,40)),m*.036,False)
    b.add_random_craters(1+int(2*s.feature_density),(m*.012,m*.028),.68,.20); b.add_detail(2*s.detail,m*.060); _repair_connectivity(b,38,.96,2,m*.008,m*.014); return b.finalize(center_height=1120,preserve_flats=True)


PLANETARY_RECIPES: Dict[str, Callable[[GeneratorSettings], HG2Map]] = {
    "Pluto Basin": pluto_basin,
    "Venus Shield": venus_shield,
    "Lunar Catena": lunar_catena,
    "Mars Rift": mars_rift,
    "Callisto Craterlands": callisto_craterlands,
    "Titan Basin Network": titan_basin_network,
    "Europa Fracture Plains": europa_fracture_plains,
}
