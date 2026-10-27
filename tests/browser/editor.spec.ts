import {test,expect} from '@playwright/test';
import type { Locator } from '@playwright/test';

test.beforeEach(async({page})=>{await page.goto('/');await expect(page.getByRole('button',{name:'Shapes compatible'})).toBeVisible();});

test('one canvas composes sampling, n-grams and a recurrent transformer',async({page})=>{
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
  await expect(page.locator('.return-path')).toHaveCount(1);
  await page.getByRole('button',{name:'Open loop body'}).click();
  await expect(page.locator('.breadcrumbs')).toContainText('Iterative denoising');
  await expect(page.locator('.sampling-preview')).toBeVisible();
  await page.getByRole('button',{name:'Open block'}).click();
  await expect(page.locator('.embedding-grid')).toHaveCount(2);
  await expect(page.locator('.ngram-preview')).toHaveCount(1);
  await expect(page.locator('.return-path')).toHaveCount(1);
  await page.getByRole('button',{name:'Open loop body'}).click();
  await expect(page.locator('.attention-matrix')).toHaveCount(1);
  await expect(page.locator('.residual-preview')).toHaveCount(2);
  await expect(page.getByRole('button',{name:'Shapes compatible'})).toBeVisible();
  expect(errors).toEqual([]);
  await page.screenshot({path:'data/editor-transformer.png'});
});

test('edit properties and show an actionable nested shape mismatch',async({page})=>{
  await page.getByRole('button',{name:'Open loop body'}).click();
  await page.getByRole('button',{name:'Open block'}).click();
  // Fit a target node before interacting so the inspector test is independent of zoom.
  const memory=page.locator('.react-flow__node[data-id="memory"]');
  await memory.click();
  await page.getByRole('textbox',{name:'width',exact:true}).fill('768');
  await page.getByRole('textbox',{name:'width',exact:true}).press('Enter');
  await expect(page.locator('.validation-bar')).toContainText('graph issues');
  await page.locator('.validation-bar button').click();
  await expect(page.locator('.checks-drawer')).toContainText('768');
  await expect(page.locator('.react-flow__node[data-id="fusion"] .arch-node')).toHaveClass(/has-error/);
  await page.getByTitle('Undo (Ctrl+Z)').click();
  await expect(page.getByRole('button',{name:'Shapes compatible'})).toBeVisible();
});

test('reference graphs load, expand and export',async({page})=>{
  await page.getByRole('button',{name:'Models',exact:true}).click();
  for(const name of ['Needle 1','Needle 2','Needle 3','Thimble']){
    await page.locator('.model-item').filter({hasText:name}).click();
    await expect(page.locator('.project-title')).toContainText(name);
    await expect(page.getByRole('button',{name:'Shapes compatible'})).toBeVisible();
    await page.getByRole('button',{name:'References',exact:true}).click();
    await expect(page.locator('.references-view article')).not.toHaveCount(0);
    await page.getByRole('button',{name:'Architecture',exact:true}).click();
  }
  const download=page.waitForEvent('download');
  await page.getByRole('button',{name:'Export PyTorch'}).click();
  const file=await download;expect(file.suggestedFilename()).toContain('.zip');
});

test('save, reload and import a project',async({page})=>{
  await page.getByRole('button',{name:'Save',exact:true}).click();
  await expect(page.getByRole('status')).toContainText('saved');
  const savedId=await page.evaluate(()=>JSON.parse(localStorage.getItem('weave.project')!).id);
  await page.reload();await expect(page.locator('.project-title')).toContainText('Composed language model');
  await page.getByRole('button',{name:'PyTorch code',exact:true}).click();
  await expect(page.locator('.code-view')).toContainText('class ComposedLanguageModel(nn.Module)');
  await expect(page.locator('.code-view')).toContainText('for step_index in range(');
  await expect(page.locator('.code-view')).not.toContainText('make_module');
  await expect(page.locator('.code-view')).not.toContainText('ModuleDict');
  const templates=await(await page.request.get('/api/templates')).json();
  await page.locator('input[type=file]').first().setInputFiles({name:'project.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(templates[2]))});
  await expect(page.locator('.project-title')).toContainText('Looped transformer');
  await page.request.delete('/api/projects/'+savedId);
});

test('tokenizer upload detects capacity, updates dimensions and previews token IDs',async({page})=>{
  test.setTimeout(60000);
  await page.locator('.tokenizer-chip').click();
  const bundle={version:'1.0',truncation:null,padding:null,added_tokens:[{id:3,content:'[MASK]',single_word:false,lstrip:false,rstrip:false,normalized:false,special:true}],normalizer:null,pre_tokenizer:{type:'Whitespace'},post_processor:null,decoder:null,model:{type:'WordLevel',vocab:{'[UNK]':0,'hello':1,'world':2,'[MASK]':3},unk_token:'[UNK]'}};
  await page.locator('input[type=file]').last().setInputFiles({name:'tokenizer.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(bundle))});
  await expect(page.locator('.tokenizer-result')).toContainText('Embedding capacity',{timeout:45000});
  await expect(page.locator('.token-stats')).toContainText('4');
  await page.locator('.tokenizer-result textarea').fill('hello world');
  await page.getByRole('button',{name:'Tokenize sample'}).click();
  await expect(page.locator('.token-pills')).toContainText('hello');
  await expect(page.locator('.preview-shape')).toContainText('[1, 2]');
  await page.getByRole('button',{name:'Close dialog'}).click();
  await expect(page.locator('.tokenizer-chip')).toContainText('4 vocab');
  await expect(page.getByRole('button',{name:'Shapes compatible'})).toBeVisible();
});

test('custom Python is editable and explicitly checked by contract',async({page})=>{
  await page.getByRole('textbox',{name:'Search blocks'}).fill('Python');
  await page.locator('.library-item').filter({hasText:'Python block'}).click();
  await expect(page.getByRole('textbox',{name:'Python source'})).toContainText('class CustomBlock');
  await expect(page.locator('.shape-card')).toContainText('Unresolved');
});

test('desktop and narrow viewport have no horizontal overflow',async({page})=>{
  await page.screenshot({path:'data/editor-overview.png'});
  await page.setViewportSize({width:900,height:760});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBe(true);
  await expect(page.locator('.inspector')).toBeVisible();
  await page.setViewportSize({width:390,height:844});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBe(true);
  await expect(page.getByRole('button',{name:'Export PyTorch'})).toBeVisible();
});

test('create, connect and validate a graph using editable ports',async({page})=>{
  await page.getByRole('button',{name:'Models',exact:true}).click();
  await page.locator('.model-item').filter({hasText:'Blank project'}).click();
  await page.getByRole('button',{name:'Blocks',exact:true}).click();
  await page.locator('.library-item').filter({hasText:'Token IDs → learned vectors'}).click();
  await page.getByTitle('Fit graph').click();
  await page.waitForTimeout(350);
  const embedding=page.locator('.react-flow__node').filter({has:page.locator('.node-title').filter({hasText:/^Embedding$/})});
  async function connect(source:Locator,target:Locator){
    const a=await source.boundingBox(),b=await target.boundingBox();
    if (!a || !b) throw new Error("Connection handles must be visible");
    await page.mouse.move(a.x+a.width/2,a.y+a.height/2);await page.mouse.down();
    await page.mouse.move(b.x+b.width/2,b.y+b.height/2,{steps:15});await page.mouse.up();
  }
  await connect(page.locator('.react-flow__node[data-id="ids"] .react-flow__handle.source'),embedding.locator('.react-flow__handle.target'));
  await connect(embedding.locator('.react-flow__handle.source'),page.locator('.react-flow__node[data-id="out"] .react-flow__handle.target'));
  await expect(page.getByRole('button',{name:'Shapes compatible'})).toBeVisible();
  await expect(page.locator('.react-flow__edge')).toHaveCount(2);
});

test('package a selection as an editable reusable block',async({page})=>{
  await page.locator('.react-flow__node[data-id="mask"]').click();
  await page.getByTitle('Create reusable block (Ctrl+G)').click();
  await expect(page.getByRole('textbox',{name:'Block name'})).toHaveValue('Custom block');
  await expect(page.getByRole('button',{name:'Shapes compatible'})).toBeVisible();
  await page.getByRole('button',{name:'Save to block library'}).click();
  await page.getByRole('button',{name:'Open internal graph'}).click();
  await expect(page.locator('.node-title').filter({hasText:'Append masked tokens'})).toBeVisible();
  await expect(page.getByRole('button',{name:'Shapes compatible'})).toBeVisible();
});

test('expand attention into editable Q/K/V computation',async({page})=>{
  await page.getByRole('button',{name:'Open loop body'}).click();
  await page.getByRole('button',{name:'Open block'}).click();
  await page.getByRole('button',{name:'Open loop body'}).click();
  await page.locator('.react-flow__node[data-id="attn"]').click();
  await page.getByRole('button',{name:'Edit attention internals'}).click();
  await expect(page.locator('.node-title').filter({hasText:'Q projection'})).toBeVisible();
  await expect(page.locator('.node-title').filter({hasText:'Scaled dot-product attention'})).toBeVisible();
  await expect(page.getByRole('button',{name:'Shapes compatible'})).toBeVisible();
  await page.screenshot({path:'data/editor-attention-internals.png'});
});
