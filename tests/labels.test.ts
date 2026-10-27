import test from 'node:test';
import assert from 'node:assert/strict';
import {placeLabels,rectanglesOverlap} from '../frontend/src/labels.ts';
import type {LabelRequest} from '../frontend/src/labels.ts';
import type {Rectangle} from '../frontend/src/routing.ts';
const request=(id:string,width=140):LabelRequest=>({id,points:[{x:0,y:0},{x:350,y:0}],width,height:26,priority:1,preferred:{x:175,y:0}});

test('a clear centre does not allow the text box to cover a card',()=>{
  const card:Rectangle={id:'card',x:200,y:-10,width:200,height:150},label=request('label');
  const placed=placeLabels([label],[card]).get('label');assert(placed);assert(!rectanglesOverlap(placed.box,card,6));assert.notDeepEqual(placed.center,label.preferred);
});
test('label boxes on shared routes do not overlap each other',()=>{
  const placements=[...placeLabels([request('a'),request('b'),request('c')],[]).values()];assert.equal(placements.length,3);
  for(let i=0;i<placements.length;i++)for(let j=i+1;j<placements.length;j++)assert(!rectanglesOverlap(placements[i].box,placements[j].box,6));
});
test('selected labels get first choice and results are deterministic',()=>{
  const a=request('a'),b={...request('b'),priority:0};const first=placeLabels([a,b],[]),second=placeLabels([b,a],[]);assert.deepEqual(first,second);assert.deepEqual(first.get('b')?.center,b.preferred);
});
test('a fully occupied fixed drawing never forces an overlapping label',()=>{
  const occupied={id:'occupied',x:-1000,y:-1000,width:2000,height:2000};assert.equal(placeLabels([request('label')],[occupied]).size,0);
});
test('large user-supplied labels are tested at their full dimensions',()=>{
  const nodes=[{id:'a',x:0,y:0,width:250,height:200},{id:'b',x:500,y:0,width:250,height:200}];const long={...request('long',440),points:[{x:250,y:-55},{x:750,y:-55}],height:32};const p=placeLabels([long],nodes).get('long');assert(p);assert.equal(p.box.width,440);assert(nodes.every(n=>!rectanglesOverlap(p.box,n,6)));
});
