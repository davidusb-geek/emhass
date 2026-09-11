// DOM lifecycle/geometry-input tests, not browser rendering tests.
// Run from repository root: node --test tests/ui/sticky-lifecycle.test.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('src/emhass/static/script.js','utf8');
function target() {
  const events = new Map();
  return { events, addEventListener: (k,f) => { if(!events.has(k)) events.set(k,new Set());events.get(k).add(f); },
    removeEventListener:(k,f)=>events.get(k)?.delete(f), emit:k=>[...(events.get(k)||[])].forEach(f=>f()),
    count:k=>events.get(k)?.size||0 };
}
function setup({width=1000, left=50, tableWidth=2000, margin=0, scroll=0, top=-100}={}) {
  const window=target(), wrappers=[], frames=new Map(); let nextFrame=1;
  const rect={width:tableWidth,left:left+margin-scroll,bottom:500};
  const headRect={top};
  const cells=[200,300,1500].map(width=>({getBoundingClientRect:()=>({width})}));
  const container=Object.assign(target(),{clientLeft:0,clientWidth:width,scrollLeft:scroll,getBoundingClientRect:()=>({left})});
  const head={getBoundingClientRect:()=>headRect,querySelectorAll:()=>cells,cloneNode:()=>({})};
  const table={className:'mystyle',querySelector:()=>head,closest:()=>container,getBoundingClientRect:()=>rect};
  let tables=[table];
  const document={body:{appendChild:w=>wrappers.push(w)},
    querySelectorAll:s=>s==='.sticky-header-wrapper'?[...wrappers]:tables,
    createElement:tag=>{
      if(tag==='div'){const w={style:{},scrollLeft:0,appendChild:c=>w.clone=c,remove:()=>{const i=wrappers.indexOf(w);if(i>=0)wrappers.splice(i,1);}};return w;}
      const ths=cells.map(()=>({style:{}}));
      return {style:{},appendChild:()=>{},querySelectorAll:()=>ths,querySelector:()=>ths[0]};
    }};
  const ctx={window,document,requestAnimationFrame:f=>{const id=nextFrame++;frames.set(id,f);return id;},cancelAnimationFrame:id=>frames.delete(id)};
  vm.createContext(ctx);vm.runInContext(source,ctx);
  return {ctx,window,wrappers,frames,container,rect,headRect,detach:()=>{tables=[];},flush:()=>{for(const [id,f]of [...frames]){frames.delete(id);f();}}};
}
test('initially scrolled table synchronizes before a new scroll event',()=>{
  const s=setup({scroll:350});s.ctx.initStickyTables();s.flush();
  assert.equal(s.wrappers[0].scrollLeft,350);
  assert.equal(s.wrappers[0].clone.querySelector().style.transform,'translateX(350px)');
});
test('sticky header is visible immediately if the live header is above viewport',()=>{
  const s=setup();s.ctx.initStickyTables();s.flush();assert.equal(s.wrappers[0].style.visibility,'visible');
});
test('sticky header remains hidden when original header is visible',()=>{
  const s=setup({top:120});s.ctx.initStickyTables();s.flush();assert.equal(s.wrappers[0].style.visibility || 'hidden','hidden');
});
test('header is hidden after the table exits viewport',()=>{
  const s=setup();s.ctx.initStickyTables();s.flush();s.rect.bottom=-1;s.window.emit('scroll');
  assert.equal(s.wrappers[0].style.visibility || 'hidden','hidden');
});
test('measured column widths are copied to cloned cells',()=>{
  const s=setup();s.ctx.initStickyTables();s.flush();const th=s.wrappers[0].clone.querySelector();
  assert.equal(th.style.width,'200px');
});
test('centered narrow table carries its horizontal offset into clone',()=>{
  const s=setup({tableWidth:600,margin:200});s.ctx.initStickyTables();s.flush();
  assert.equal(s.wrappers[0].clone.style.margin,'0 0 0 200px');
});
test('reinitializing does not accumulate resize or scroll listeners',()=>{
  const s=setup();for(let i=0;i<5;i++){s.ctx.initStickyTables();s.flush();}
  assert.equal(s.wrappers.length,1);assert.equal(s.window.count('resize'),1);
  assert.equal(s.window.count('scroll'),1);assert.equal(s.container.count('scroll'),1);
});
test('removed original tables still have their listeners cleaned up',()=>{
  const s=setup();s.ctx.initStickyTables();s.flush();s.detach();s.ctx.initStickyTables();s.flush();
  assert.equal(s.wrappers.length,0);assert.equal(s.window.count('resize'),0);
  assert.equal(s.window.count('scroll'),0);assert.equal(s.container.count('scroll'),0);
});
test('pending geometry animation is cancelled on reinitialization',()=>{
  const s=setup();s.ctx.initStickyTables();s.ctx.initStickyTables();assert.equal(s.frames.size,1);s.flush();
});
test('resize re-synchronizes scroll and visibility',()=>{
  const s=setup();s.ctx.initStickyTables();s.flush();s.container.scrollLeft=125;s.headRect.top=100;
  s.window.emit('resize');assert.equal(s.wrappers[0].scrollLeft,125);assert.equal(s.wrappers[0].style.visibility || 'hidden','hidden');
});
