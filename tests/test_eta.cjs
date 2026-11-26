const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const context={location:{protocol:'file:'}};vm.createContext(context);
vm.runInContext(fs.readFileSync('src/video_privacy/assets/simple.js','utf8'),context);
const estimate=context.estimateProgress;
assert.equal(estimate([{count:0,time:0}],100,0).seconds,null);
assert.equal(estimate([{count:0,time:0},{count:10,time:20000}],100,20000).seconds,180);
assert.equal(estimate([{count:0,time:0},{count:10,time:20000},{count:10,time:90000}],100,90000).stalled,true);
assert.equal(estimate([{count:0,time:0},{count:10,time:20000}],10,20000).seconds,null);
console.log('ETA cold start, throughput, stall and completion checks passed');
