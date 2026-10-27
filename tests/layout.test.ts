import test from 'node:test';
import assert from 'node:assert/strict';
import ELK from 'elkjs/lib/elk.bundled.js';
import {buildLayoutGraph,applyLayout,layoutSignature} from '../frontend/src/layout-core.ts';
import type {Graph,GraphNode} from '../frontend/src/types.ts';
import {rectanglesOverlap} from '../frontend/src/labels.ts';
import {nodeRectangles} from '../frontend/src/routing.ts';
const node=(id:string,x:number,y:number):GraphNode=>({id,type:'architecture',position:{x,y},measured:{width:248,height:200},data:{kind:'linear',label:id,params:{out:128}}});
const graph:Graph={nodes:[node('a',0,0),node('b',0,300),node('c',700,0),node('d',700,300)],edges:[{id:'ad',source:'a',target:'d',targetHandle:'a'},{id:'bc',source:'b',target:'c',targetHandle:'a'}]};

test('ELK reorders a crossed pair and reserves full label boxes',async()=>{
  const engine=new ELK();const result=await engine.layout(buildLayoutGraph(graph,new Map(graph.edges.map(e=>[e.id,{text:'B × T × 128',width:160,height:28}]))));const next=applyLayout(graph,result),nodes=new Map(next.nodes.map(n=>[n.id,n]));
  assert((nodes.get('a')!.position.y-nodes.get('b')!.position.y)*(nodes.get('d')!.position.y-nodes.get('c')!.position.y)>0);
  const rectangles=nodeRectangles(next.nodes);
  for(const edge of result.edges||[]){const label=edge.labels?.[0];assert(label?.x!==undefined&&label.y!==undefined);const box={id:edge.id,x:label.x,y:label.y,width:label.width||0,height:label.height||0};assert(rectangles.every(n=>!rectanglesOverlap(box,n)));assert(next.layout?.routes[edge.id]?.points.length);}
  assert.equal(next.layout?.signature,layoutSignature(next));assert.deepEqual(next.edges,graph.edges);
});
test('moving any card invalidates cached routes, while selection does not',async()=>{
  const next=applyLayout(graph,await new ELK().layout(buildLayoutGraph(graph,new Map())));const original=layoutSignature(next);next.nodes[0].selected=true;assert.equal(layoutSignature(next),original);next.nodes[2].position.x+=50;assert.notEqual(layoutSignature(next),original);
});
