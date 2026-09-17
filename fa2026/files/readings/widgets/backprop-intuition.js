/* Offline intuition labs. Pure computations are also exported for numerical checks. */
(function () {
    'use strict';
    function network(p) {
        var z = [p.w00*p.x0+p.w01*p.x1+p.b0, p.w10*p.x0+p.w11*p.x1+p.b1];
        var h = z.map(function(v) { return p.activation === 'tanh' ? Math.tanh(v) : Math.max(0,v); });
        var pred = p.u0*h[0]+p.u1*h[1]+p.c, q = pred-p.target;
        var gh = [q*p.u0,q*p.u1];
        var delta = gh.map(function(v,i) { return v*(p.activation === 'tanh' ? 1-h[i]*h[i] : z[i]>0 ? 1 : 0); });
        var paths = [[p.w00*delta[0],p.w10*delta[1]], [p.w01*delta[0],p.w11*delta[1]]];
        return {z:z,h:h,pred:pred,loss:0.5*q*q,q:q,gh:gh,delta:delta,paths:paths,
            grad:{w00:delta[0]*p.x0,w01:delta[0]*p.x1,w10:delta[1]*p.x0,w11:delta[1]*p.x1,
                b0:delta[0],b1:delta[1],u0:q*h[0],u1:q*h[1],c:q,
                x0:paths[0][0]+paths[0][1],x1:paths[1][0]+paths[1][1]}};
    }
    function softmax(logits,target) {
        var m=Math.max.apply(null,logits), e=logits.map(function(v){return Math.exp(v-m);});
        var sum=e.reduce(function(a,b){return a+b;},0), p=e.map(function(v){return v/sum;});
        return {p:p,loss:(m-logits[target])+Math.log(sum),grad:p.map(function(v,i){return v-(i===target?1:0);})};
    }
    function inspector(x,requires,retain,detach,noGrad,ran) {
        var hRequires=requires&&!noGrad, connected=hRequires&&!detach, flows=connected&&ran;
        return {hRequires:hRequires,connected:connected,flows:flows,
            xGrad:flows?9*x:null,hGrad:flows&&retain?3*x:null,upstream:flows?3*x:null};
    }
    var api={network:network,softmax:softmax,inspector:inspector};
    if (typeof module !== 'undefined' && module.exports) module.exports=api;
    if (typeof document === 'undefined') return;
    function f(v) { return Math.abs(v)<0.00005?'0':Number(v.toFixed(4)).toString(); }
    function bind(root,sel,event,fn) { root.querySelector(sel).addEventListener(event,fn); }
    function num(label,key,value,min,max,step) {
        return '<label>'+label+' <input aria-label="'+label+'" data-key="'+key+'" type="number" min="'+min+'" max="'+max+'" step="'+step+'" value="'+value+'"></label>';
    }
    function cell(v) {return '<td>'+v+'</td>';}
    function initNetwork(root) {
        var defaults={x0:1,x1:2,target:1,w00:1,w01:0,w10:0,w11:-1,b0:0,b1:1,u0:2,u1:3,c:0,activation:'relu'};
        var p=Object.assign({},defaults), stage=0;
        root.innerHTML='<div class="bi-controls">'+num('Input x₁','x0',1,-4,4,0.25)+num('Input x₂','x1',2,-4,4,0.25)+num('Target t','target',1,-4,4,0.25)+
            '<label>Activation <select data-activation><option value="relu">ReLU</option><option value="tanh">tanh</option></select></label></div>'+
            '<details class="bi-parameters"><summary>Edit weights and biases</summary><div class="bi-controls">'+
            ['w00','w01','w10','w11','b0','b1','u0','u1','c'].map(function(k,i){return num(['W₁₁','W₁₂','W₂₁','W₂₂','b₁','b₂','u₁','u₂','c'][i],k,p[k],-4,4,0.25);}).join('')+'</div></details>'+
            '<p class="w-note">Predict the next values before revealing them. Editing any input starts a fresh pass. ReLU uses derivative 0 at zero.</p>'+
            '<div class="w-row"><button type="button" class="w-btn" data-forward>Forward step</button><button type="button" class="w-btn" data-backward>Backward step</button><button type="button" class="w-btn" data-reset>Reset example</button></div>'+
            '<div class="bi-graph" data-graph></div><p class="bi-status" data-status role="status" aria-live="polite"></p><div data-calculation></div>';
        function render() {
            var n=network(p), labels=['Inputs ready','z = Wx + b','h = activation(z)','ŷ = uᵀh + c','L = ½(ŷ − t)²','q = ∂L/∂ŷ = ŷ − t','Output parameters and hidden sensitivities','δ = (q u) ⊙ activation′(z)','Hidden parameters and input gradients'];
            var names=['x₁','x₂','z₁','z₂','h₁','h₂','ŷ','L'];
            var values=[p.x0,p.x1,n.z[0],n.z[1],n.h[0],n.h[1],n.pred,n.loss];
            var reveal=[0,0,1,1,2,2,3,4], gradAt=[8,8,7,7,6,6,5,5];
            var grads=[n.grad.x0,n.grad.x1,n.delta[0],n.delta[1],n.gh[0],n.gh[1],n.q,1];
            var pos=[[52,60],[52,170],[180,60],[180,170],[308,60],[308,170],[436,115],[564,115]];
            var edges=[[0,2],[0,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,6],[6,7]];
            var svg='<svg viewBox="0 0 620 235" role="img" aria-label="Forward values and loss gradients through a two-hidden-unit network"><defs><marker id="bi-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#6b7480"/></marker></defs>';
            edges.forEach(function(e){var a=pos[e[0]],b=pos[e[1]];svg+='<line x1="'+(a[0]+38)+'" y1="'+a[1]+'" x2="'+(b[0]-40)+'" y2="'+b[1]+'" stroke="#9aa9bc" marker-end="url(#bi-arrow)"/>';});
            pos.forEach(function(a,i){svg+='<g><rect x="'+(a[0]-38)+'" y="'+(a[1]-29)+'" width="76" height="67" rx="8" fill="'+(stage>=gradAt[i]?'#edf7f1':'#f2f5fa')+'" stroke="#cad3df"/><text x="'+a[0]+'" y="'+(a[1]-12)+'" text-anchor="middle">'+names[i]+'</text><text x="'+a[0]+'" y="'+(a[1]+6)+'" text-anchor="middle">'+(stage>=reveal[i]?f(values[i]):'?')+'</text><text x="'+a[0]+'" y="'+(a[1]+26)+'" text-anchor="middle" fill="#226345">g: '+(stage>=gradAt[i]?f(grads[i]):'—')+'</text></g>';});
            root.querySelector('[data-graph]').innerHTML=svg+'</svg><p class="w-note">Arrows show forward dependencies. Green nodes have revealed loss gradients (g); “—” means not revealed yet.</p>';
            root.querySelector('[data-status]').textContent='Step '+stage+'/8 · '+labels[stage];
            root.querySelector('[data-forward]').disabled=stage>=4;
            root.querySelector('[data-backward]').disabled=stage<4||stage>=8;
            var lines=[
                'Use the handout’s worked example, or edit the inputs and parameters.',
                'z₁ = '+f(p.w00)+' × '+f(p.x0)+' + '+f(p.w01)+' × '+f(p.x1)+' + '+f(p.b0)+' = '+f(n.z[0])+'; z₂ = '+f(p.w10)+' × '+f(p.x0)+' + '+f(p.w11)+' × '+f(p.x1)+' + '+f(p.b1)+' = '+f(n.z[1]),
                'h = ['+n.h.map(f).join(', ')+']; each activation acts on its own preactivation.',
                'ŷ = '+f(p.u0)+' × '+f(n.h[0])+' + '+f(p.u1)+' × '+f(n.h[1])+' + '+f(p.c)+' = '+f(n.pred),
                'Loss = ½ × ('+f(n.pred)+' − '+f(p.target)+')² = '+f(n.loss)+'. Now seed ∂L/∂L = 1 and work backward.',
                'Incoming gradient 1 × local derivative (ŷ − t) = '+f(n.q)+'.',
                '∇u = q h = ['+[n.grad.u0,n.grad.u1].map(f).join(', ')+']; ∂L/∂c = '+f(n.q)+'. Pass gₕ = q u = ['+n.gh.map(f).join(', ')+'] to the hidden units.',
                'δ = gₕ ⊙ activation′(z) = ['+n.delta.map(f).join(', ')+']. An inactive ReLU blocks that path.',
                'Each input receives two path contributions. Add them; weights have not been updated.'
            ];
            var detail='<p class="bi-equation">'+lines[stage]+'</p>';
            if(stage===8) detail+='<div class="tbl-scroll"><table class="tbl"><thead><tr><th>Gradient</th><th>Computation / value</th></tr></thead><tbody><tr>'+cell('∇W = δ xᵀ')+cell('[['+[n.grad.w00,n.grad.w01].map(f).join(', ')+'], ['+[n.grad.w10,n.grad.w11].map(f).join(', ')+']]')+'</tr><tr>'+cell('∇b = δ')+cell('['+n.delta.map(f).join(', ')+']')+'</tr>'+[0,1].map(function(i){return '<tr>'+cell('∂L/∂x'+(i+1))+cell(n.paths[i].map(f).join(' + ')+' = '+f(n.grad['x'+i]))+'</tr>';}).join('')+'</tbody></table></div>';
            root.querySelector('[data-calculation]').innerHTML=detail;
        }
        root.querySelectorAll('input[data-key]').forEach(function(el){el.addEventListener('input',function(){if(el.value===''||!el.validity.valid)return;p[el.dataset.key]=Number(el.value);stage=0;render();});});
        bind(root,'[data-activation]','change',function(e){p.activation=e.target.value;stage=0;render();});
        bind(root,'[data-forward]','click',function(){if(stage<4)stage++;render();});
        bind(root,'[data-backward]','click',function(){if(stage>=4&&stage<8)stage++;render();});
        bind(root,'[data-reset]','click',function(){p=Object.assign({},defaults);stage=0;root.querySelectorAll('input[data-key]').forEach(function(el){el.value=p[el.dataset.key];});root.querySelector('[data-activation]').value=p.activation;render();});
        render();
    }
    function initSoftmax(root) {
        var logits=[2,1,0],target=0,shift=0;
        root.innerHTML='<div class="bi-controls">'+[0,1,2].map(function(i){return '<label>Class '+(i+1)+' logit <output data-logit-label="'+i+'"></output><input type="range" aria-label="Class '+(i+1)+' logit" data-logit="'+i+'" min="-8" max="8" step="0.1" value="'+logits[i]+'"></label>';}).join('')+'</div>'+
            '<div class="bi-controls"><label>Correct class <select data-target><option value="0">Class 1</option><option value="1">Class 2</option><option value="2">Class 3</option></select></label><label>Common logit shift <output data-shift-label>0</output><input type="range" aria-label="Common logit shift" data-shift min="-1000" max="1000" step="10" value="0"></label></div>'+
            '<div data-bars></div><p data-stats class="bi-status" role="status" aria-live="polite"></p><div class="w-row"><button class="w-btn" type="button" data-step>One logit gradient step (η = 0.5)</button><button class="w-btn" type="button" data-reset>Reset</button></div><p class="w-note">Predict which probability will rise. This step updates logits directly; training a network updates shared parameters through the chain rule. The common shift changes all three logits equally.</p>';
        function render(){
            var n=softmax(logits.map(function(v){return v+shift;}),target);
            root.querySelectorAll('[data-logit]').forEach(function(el,i){el.min=Math.floor(Math.min(-8,logits[i]));el.max=Math.ceil(Math.max(8,logits[i]));el.value=logits[i];root.querySelector('[data-logit-label="'+i+'"]').textContent=f(logits[i]+shift);});
            root.querySelector('[data-shift-label]').textContent=shift;
            var svg='<svg viewBox="0 0 600 175" role="img" aria-label="Class probabilities and signed logit gradients"><text x="125" y="20">Probability (0 to 1)</text><text x="363" y="20">Gradient p − y (−1 to +1)</text><line x1="445" y1="30" x2="445" y2="166" stroke="#8190a4"/>';
            [0,1,2].forEach(function(i){var y=45+i*43,g=n.grad[i];svg+='<text x="8" y="'+(y+15)+'">Class '+(i+1)+(i===target?' ✓':'')+'</text><rect x="100" y="'+y+'" width="180" height="22" fill="#edf1f6"/><rect x="100" y="'+y+'" width="'+(n.p[i]*180)+'" height="22" fill="#528dca"/><text x="290" y="'+(y+15)+'">'+f(n.p[i])+'</text><rect x="'+(445+Math.min(0,g)*70)+'" y="'+y+'" width="'+(Math.abs(g)*70)+'" height="22" fill="'+(g<0?'#b02a55':'#226345')+'"/><text x="525" y="'+(y+15)+'">'+f(g)+'</text>';});
            root.querySelector('[data-bars]').innerHTML='<div class="bi-graph">'+svg+'</svg></div>';
            root.querySelector('[data-stats]').textContent='Cross-entropy: '+f(n.loss)+' nats · Sum of gradients: '+f(n.grad.reduce(function(a,b){return a+b;},0));
        }
        root.querySelectorAll('[data-logit]').forEach(function(el,i){el.addEventListener('input',function(){logits[i]=Number(el.value);render();});});
        bind(root,'[data-target]','change',function(e){target=Number(e.target.value);render();});
        bind(root,'[data-shift]','input',function(e){shift=Number(e.target.value);render();});
        bind(root,'[data-step]','click',function(){var g=softmax(logits,target).grad;logits=logits.map(function(v,i){return v-0.5*g[i];});render();});
        bind(root,'[data-reset]','click',function(){logits=[2,1,0];target=0;shift=0;root.querySelector('[data-target]').value='0';root.querySelector('[data-shift]').value='0';render();});
        render();
    }
    function initInspector(root) {
        var ran=false, inputX=2;
        root.innerHTML='<p class="bi-equation">x → h = 3x → t → L = ½t²</p><p class="w-note">Toggle how h and t are constructed, then call backward. This is a simulation of the displayed PyTorch example. Each edit creates fresh tensors with empty gradient buffers.</p><div class="bi-controls">'+num('Input x','x',2,-3,3,0.5)+'</div><div class="bi-toggles">'+
            [['requires','x.requires_grad',true],['retain','h.retain_grad()',false],['detach','t = h.detach() (otherwise t = h)',false],['no-grad','Build h = 3*x inside no_grad()',false]].map(function(a){return '<label><input type="checkbox" data-'+a[0]+(a[2]?' checked':'')+'> '+a[1]+'</label>';}).join('')+'</div><div class="w-row"><button class="w-btn" type="button" data-run>Run backward()</button><button class="w-btn" type="button" data-reset>Reset</button></div><p class="bi-status" role="status" aria-live="polite" data-status></p><div data-table></div><pre class="bi-code" data-code></pre>';
        function checked(k){return root.querySelector('[data-'+k+']').checked;}
        function render(){
            var x=inputX, req=checked('requires'), detach=checked('detach'), noGrad=checked('no-grad');
            var retainEl=root.querySelector('[data-retain]');retainEl.disabled=!req||noGrad;if(retainEl.disabled)retainEl.checked=false;
            var retain=retainEl.checked,n=inspector(x,req,retain,detach,noGrad,ran);
            var status=!ran?'Predict: where will gradients flow, and which buffers will be filled?':!n.connected?'Backward raises an error: this loss does not require gradients. No gradients are stored.':x===0?'Backward succeeded. The derivative is zero here; zero is different from None.':'Backward succeeded. A flowing derivative is stored only where requested.';
            if(retainEl.disabled)status+=' retain_grad is unavailable because h does not require gradients.';
            root.querySelector('[data-status]').textContent=status;
            var rows=[['x',x,req,'leaf',n.flows?f(n.xGrad):'—',n.xGrad===null?'None':f(n.xGrad)],['h',3*x,n.hRequires,n.hRequires?'non-leaf':'leaf',n.flows?f(n.upstream):'—',n.hGrad===null?'None':f(n.hGrad)],['L',4.5*x*x,n.connected,n.connected?'non-leaf':'leaf',n.flows?'1 (seed)':'—','None']];
            root.querySelector('[data-table]').innerHTML='<div class="tbl-scroll"><table class="tbl"><thead><tr><th>Tensor</th><th>Value</th><th>requires_grad</th><th>Kind</th><th>Backward signal</th><th>Stored .grad</th></tr></thead><tbody>'+rows.map(function(row){return '<tr>'+row.map(cell).join('')+'</tr>';}).join('')+'</tbody></table></div><p class="w-note">“—” means no backward signal was computed in this run. L is non-leaf when tracked, so its .grad stays None even though the backward seed is 1. '+(detach?'The detached t shares h’s storage but cuts the path from L to h.':'t and h are the same tensor.')+'</p>';
            root.querySelector('[data-code]').textContent='x = torch.tensor('+x.toFixed(1)+', requires_grad='+ (req?'True':'False')+')\n'+(noGrad?'with torch.no_grad():\n    h = 3*x':'h = 3*x')+'\n'+(retain?'h.retain_grad()\n':'')+'t = '+(detach?'h.detach()':'h')+'\nloss = 0.5*t.square()\nloss.backward()';
            root.querySelector('[data-run]').disabled=ran;
        }
        root.querySelectorAll('input').forEach(function(el){el.addEventListener('input',function(){if(el.type==='number'){if(el.value===''||!el.validity.valid)return;inputX=Number(el.value);}ran=false;render();});});
        bind(root,'[data-run]','click',function(){ran=true;render();});
        bind(root,'[data-reset]','click',function(){inputX=2;root.querySelector('[data-key="x"]').value=2;['requires','retain','detach','no-grad'].forEach(function(k){root.querySelector('[data-'+k+']').checked=k==='requires';});ran=false;render();});
        render();
    }
    [['w-backprop-stepper',initNetwork],['w-softmax-playground',initSoftmax],['w-gradient-inspector',initInspector]].forEach(function(pair){var root=document.getElementById(pair[0]);if(root)pair[1](root);});
})();
