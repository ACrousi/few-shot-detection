// Run with node --test dinov3_ws/tests/test_images_ui.cjs
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../src/dinov3_cases/dinov3_cases/api/static/images.js'), 'utf8');

function page() {
  const elements = new Map();
  function element(id = '') {
    return {id, value: '', textContent: '', disabled: false, style: {}, children: [], options: [],
      setAttribute() {}, removeAttribute() {}, focus() { this.focused = true; },
      replaceChildren(...items) { this.children = items; this.options = items; this.replacements = (this.replacements || 0) + 1; },
      append(item) { this.children.push(item); },
      after(item) { elements.set(item.id, item); },
    };
  }
  const ids = [...source.matchAll(/el\('([^']+)'\)/g)].map(match => match[1]);
  for (const id of [...ids, 'hue', 'saturation', 'value', 'feather']) elements.set(id, element(id));
  elements.get('camera-threshold').value = '0.3';
  elements.delete('camera-delete-label');
  elements.delete('camera-capture-help'); // Reproduce a service with cached old HTML.
  const requests = [], timers = [];
  const state = {status: {camera_connected: true}, labels: ['medicine'], confirmResult: false, confirmations: []};
  const context = vm.createContext({
    document: {getElementById: id => elements.get(id), createElement: () => element()},
    confirm: text => { state.confirmations.push(text); return state.confirmResult; },
    Option: function(text, value) { return {textContent: text, value}; },
    setTimeout: fn => { timers.push(fn); }, clearTimeout() {}, URL, AbortController,
    fetch: async (url, options) => {
      requests.push({url, options});
      if (url === '/api/status' && state.error) throw Error('status unavailable');
      if (options?.method === 'DELETE' && url.startsWith('/api/images/labels/')) {
        if (state.deleteError) return {ok: false, status: 500, json: async () => ({error: 'Deletion failed'})};
        state.labels = [];
      }
      const data = url === '/api/status' ? state.status : url === '/api/images/capture' ?
        {case: 'medicine', names: []} : url === '/api/images/labels' ?
        {case: JSON.parse(options.body).case} : {cases: state.labels, images: []};
      return {ok: true, json: async () => data, blob: async () => new Blob(['preview']), headers: {get: () => '2'}};
    },
  });
  vm.runInContext(source, context);
  return {el: id => elements.get(id), requests, timers, state};
}
const settled = () => new Promise(resolve => setImmediate(resolve));

test('capture remains actionable and explains missing label and prompt', async () => {
  const p = page();
  await settled();
  assert.equal(p.el('camera-capture').disabled, false);
  await p.el('camera-capture').onclick();
  assert.match(p.el('camera-capture-help').textContent, /label/);
  assert.equal(p.el('camera-label').focused, true);
  p.el('camera-label').value = 'medicine';
  p.el('camera-label').oninput();
  await p.el('camera-capture').onclick();
  assert.match(p.el('camera-capture-help').textContent, /prompt/);
  assert.equal(p.requests.some(r => r.url === '/api/images/capture'), false);
});

test('selecting an existing gallery label allows capture without creating it again', async () => {
  const p = page();
  await settled();
  p.el('cases').value = 'medicine';
  p.el('cases').onchange();
  await settled();
  assert.equal(p.el('camera-label').value, 'medicine');
  p.el('camera-prompt').value = 'Detect box.';
  p.el('camera-prompt').oninput();
  await p.el('camera-capture').onclick();
  const request = p.requests.find(r => r.url === '/api/images/capture');
  assert.deepEqual(JSON.parse(request.options.body), {case: 'medicine', prompt: 'Detect box.', box_threshold: 0.3});
  assert.equal(p.requests.some(r => r.url === '/api/images/labels'), false);
});

test('status polling preserves a typed prompt and does not block capture on a transient status failure', async () => {
  const p = page();
  await settled();
  p.el('camera-label').value = 'medicine';
  p.el('camera-label').oninput();
  p.el('camera-prompt').value = '  red medicine bottle  ';
  assert.equal(p.el('camera-prompt').disabled, false);
  await p.timers.shift()();
  assert.equal(p.el('camera-prompt').value, '  red medicine bottle  ');
  assert.equal(p.el('camera-prompt').disabled, false);
  p.state.error = true;
  await p.timers.shift()();
  await p.el('camera-capture').onclick();
  const captures = p.requests.filter(r => r.url === '/api/images/capture');
  assert.equal(captures.length, 1);
  assert.equal(JSON.parse(captures[0].options.body).prompt, 'red medicine bottle');
});

test('an uncreated new label prevents capture and duplicate clicks are ignored while saving', async () => {
  const p = page();
  await settled();
  p.el('camera-label').value = 'new_label';
  p.el('camera-label').oninput();
  p.el('camera-prompt').value = 'Detect box.';
  await p.el('camera-capture').onclick();
  assert.equal(p.requests.some(r => r.url === '/api/images/capture'), false);
  p.el('camera-label').value = 'medicine';
  p.el('camera-label').oninput();
  const pending = p.el('camera-capture').onclick();
  assert.equal(p.el('camera-capture').disabled, true);
  await p.el('camera-capture').onclick();
  await pending;
  assert.equal(p.requests.filter(r => r.url === '/api/images/capture').length, 1);
  assert.equal(p.el('camera-capture').disabled, false);
});


test('bbox preview needs only a prompt, shows a result and clears it on prompt changes', async () => {
  const p = page();
  await settled();
  await p.el('camera-bbox-preview').onclick();
  assert.match(p.el('camera-bbox-status').textContent, /prompt/);
  assert.equal(p.requests.some(r => r.url === '/api/images/camera-preview'), false);
  p.el('camera-prompt').value = 'Detect box.';
  await p.el('camera-bbox-preview').onclick();
  const request = p.requests.find(r => r.url === '/api/images/camera-preview');
  assert.deepEqual(JSON.parse(request.options.body), {prompt: 'Detect box.', box_threshold: 0.3});
  assert.equal(p.el('camera-bbox-image').hidden, false);
  assert.match(p.el('camera-bbox-status').textContent, /2 bounding boxes/);
  assert.equal(p.requests.some(r => r.url === '/api/images/capture'), false);
  p.el('camera-prompt').oninput();
  assert.equal(p.el('camera-bbox-image').hidden, true);
});


test('score slider clears stale preview and sends the selected threshold to preview and capture', async () => {
  const p = page();
  await settled();
  p.el('camera-prompt').value = 'Detect box.';
  await p.el('camera-bbox-preview').onclick();
  p.el('camera-threshold').value = '0.75';
  p.el('camera-threshold').oninput();
  assert.equal(p.el('camera-threshold-out').textContent, '0.75');
  assert.equal(p.el('camera-bbox-image').hidden, true);
  const pending = p.el('camera-bbox-preview').onclick();
  assert.equal(p.el('camera-threshold').disabled, true);
  await pending;
  assert.equal(p.el('camera-threshold').disabled, false);
  const previews = p.requests.filter(r => r.url === '/api/images/camera-preview');
  assert.equal(JSON.parse(previews[1].options.body).box_threshold, 0.75);
  p.el('camera-label').value = 'medicine';
  p.el('camera-label').oninput();
  await p.el('camera-capture').onclick();
  const capture = p.requests.find(r => r.url === '/api/images/capture');
  assert.equal(JSON.parse(capture.options.body).box_threshold, 0.75);
  await p.timers.shift()();
  assert.equal(p.el('camera-threshold').value, '0.75');
});


test('label deletion requires confirmation and resets the deleted selection', async () => {
  const p = page();
  await settled();
  assert.equal(p.el('camera-delete-label').disabled, true);
  p.el('camera-label').value = 'medicine';
  p.el('camera-label').oninput();
  assert.equal(p.el('camera-delete-label').disabled, false);
  await p.el('camera-delete-label').onclick();
  assert.match(p.state.confirmations[0], /medicine.*entire dataset/);
  assert.equal(p.requests.some(r => r.options?.method === 'DELETE'), false);
  p.state.confirmResult = true;
  const pending = p.el('camera-delete-label').onclick();
  assert.equal(p.el('camera-delete-label').disabled, true);
  await p.el('camera-delete-label').onclick();
  await pending;
  const deletes = p.requests.filter(r => r.options?.method === 'DELETE');
  assert.equal(deletes.length, 1);
  assert.equal(deletes[0].url, '/api/images/labels/medicine');
  assert.equal(p.el('camera-label').value, '');
  assert.equal(p.el('cases').value, '');
  assert.equal(p.el('camera-delete-label').disabled, true);
  assert.match(p.el('message').textContent, /entire dataset deleted/);
});

test('failed label deletion preserves the selected dataset and shows the error', async () => {
  const p = page();
  await settled();
  p.el('camera-label').value = 'medicine';
  p.el('camera-label').oninput();
  p.state.confirmResult = true;
  p.state.deleteError = true;
  await p.el('camera-delete-label').onclick();
  assert.equal(p.el('camera-label').value, 'medicine');
  assert.equal(p.el('camera-delete-label').disabled, false);
  assert.equal(p.el('message').textContent, 'Deletion failed');
});
