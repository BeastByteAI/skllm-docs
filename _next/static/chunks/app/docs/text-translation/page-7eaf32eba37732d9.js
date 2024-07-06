(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[289,663,301,613,76,52,834,300,838,326,457,669,532,931],{8356:function(e,t,i){Promise.resolve().then(i.t.bind(i,231,23)),Promise.resolve().then(i.bind(i,4456)),Promise.resolve().then(i.bind(i,7408)),Promise.resolve().then(i.bind(i,2553)),Promise.resolve().then(i.bind(i,817))},4456:function(e,t,i){"use strict";i.d(t,{DocsHeader:function(){return r}});var n=i(7437),l=i(6463),s=i(5045);function r(e){let{title:t}=e,i=(0,l.usePathname)(),r=s.G.find(e=>e.links.find(e=>e.href===i));return t||r?(0,n.jsxs)("header",{className:"mb-9 space-y-1",children:[r&&(0,n.jsx)("p",{className:"font-display text-sm font-medium text-orange-500",children:r.title}),t&&(0,n.jsx)("h1",{className:"font-display text-3xl tracking-tight text-zinc-900 dark:text-white",children:t})]}):null}},7408:function(e,t,i){"use strict";i.d(t,{Fence:function(){return r}});var n=i(7437),l=i(2265),s=i(373);function r(e){let{children:t,language:i}=e;return(0,n.jsx)(s.y$,{code:t.trimEnd(),language:i,theme:{plain:{color:"#e4e4e7",fontStyle:"normal"},styles:[{types:["builtin","changed","keyword","function"],style:{color:"#ff7500"}},{types:["string","number","definition","boolean","class-name"],style:{color:"#fdba74"}},{types:["comment"],style:{color:"#6a9955"}},{types:["property","punctuation","operator"],style:{color:"#a1a1aa"}}]},children:e=>{let{className:t,style:i,tokens:s,getTokenProps:r}=e;return(0,n.jsx)("pre",{className:t,style:i,children:(0,n.jsx)("code",{children:s.map((e,t)=>(0,n.jsxs)(l.Fragment,{children:[e.filter(e=>!e.empty).map((e,t)=>(0,n.jsx)("span",{...r({token:e})},t)),"\n"]},t))})})}})}},2553:function(e,t,i){"use strict";i.d(t,{PrevNextLinks:function(){return d}});var n=i(7437),l=i(7138),s=i(6463),r=i(4839),o=i(5045);function a(e){return(0,n.jsx)("svg",{viewBox:"0 0 16 16","aria-hidden":"true",...e,children:(0,n.jsx)("path",{d:"m9.182 13.423-1.17-1.16 3.505-3.505H3V7.065h8.517l-3.506-3.5L9.181 2.4l5.512 5.511-5.511 5.512Z"})})}function c(e){let{title:t,href:i,dir:s="next",...o}=e;return(0,n.jsxs)("div",{...o,children:[(0,n.jsx)("dt",{className:"font-display text-sm font-medium text-zinc-900 dark:text-white",children:"next"===s?"Next":"Previous"}),(0,n.jsx)("dd",{className:"mt-1",children:(0,n.jsxs)(l.default,{href:i,className:(0,r.Z)("flex items-center gap-x-1 text-base font-semibold text-zinc-500 hover:text-zinc-600 dark:text-zinc-400 dark:hover:text-zinc-300","previous"===s&&"flex-row-reverse"),children:[t,(0,n.jsx)(a,{className:(0,r.Z)("h-4 w-4 flex-none fill-current","previous"===s&&"-scale-x-100")})]})})]})}function d(){let e=(0,s.usePathname)(),t=o.G.flatMap(e=>e.links),i=t.findIndex(t=>t.href===e),l=i>-1?t[i-1]:null,r=i>-1?t[i+1]:null;return r||l?(0,n.jsxs)("dl",{className:"mt-12 flex border-t border-zinc-200 pt-6 dark:border-zinc-800",children:[l&&(0,n.jsx)(c,{dir:"previous",...l}),r&&(0,n.jsx)(c,{className:"ml-auto text-right",...r})]}):null}},817:function(e,t,i){"use strict";i.d(t,{TableOfContents:function(){return o}});var n=i(7437),l=i(2265),s=i(7138),r=i(4839);function o(e){let{tableOfContents:t}=e,[i,o]=(0,l.useState)(t[0]?.id),a=(0,l.useCallback)(e=>e.flatMap(e=>[e.id,...e.children.map(e=>e.id)]).map(e=>{let t=document.getElementById(e);if(!t)return null;let i=parseFloat(window.getComputedStyle(t).scrollMarginTop);return{id:e,top:window.scrollY+t.getBoundingClientRect().top-i}}).filter(e=>null!==e),[]);function c(e){return e.id===i||!!e.children&&e.children.findIndex(c)>-1}return(0,l.useEffect)(()=>{if(0===t.length)return;let e=a(t);function i(){let t=window.scrollY,i=e[0].id;for(let n of e)if(t>=n.top-10)i=n.id;else break;o(i)}return window.addEventListener("scroll",i,{passive:!0}),i(),()=>{window.removeEventListener("scroll",i)}},[a,t]),(0,n.jsx)("div",{className:"hidden xl:sticky xl:top-[4.75rem] xl:-mr-6 xl:block xl:h-[calc(100vh-4.75rem)] xl:flex-none xl:overflow-y-auto xl:py-16 xl:pr-6",children:(0,n.jsx)("nav",{"aria-labelledby":"on-this-page-title",className:"w-56",children:t.length>0&&(0,n.jsxs)(n.Fragment,{children:[(0,n.jsx)("h2",{id:"on-this-page-title",className:"font-display text-sm font-medium text-zinc-900 dark:text-white",children:"On this page"}),(0,n.jsx)("ol",{role:"list",className:"mt-4 space-y-3 text-sm",children:t.map(e=>(0,n.jsxs)("li",{children:[(0,n.jsx)("h3",{children:(0,n.jsx)(s.default,{href:`#${e.id}`,className:(0,r.Z)(c(e)?"text-orange-500":"font-normal text-zinc-500 hover:text-zinc-600 dark:text-zinc-400 dark:hover:text-zinc-300"),children:e.title})}),e.children.length>0&&(0,n.jsx)("ol",{role:"list",className:"mt-2 space-y-3 pl-5 text-zinc-500 dark:text-zinc-400",children:e.children.map(e=>(0,n.jsx)("li",{children:(0,n.jsx)(s.default,{href:`#${e.id}`,className:c(e)?"text-orange-500":"hover:text-zinc-600 dark:hover:text-zinc-300",children:e.title})},e.id))})]},e.id))})]})})})}},5045:function(e,t,i){"use strict";i.d(t,{G:function(){return n}});let n=[{title:"Introduction",links:[{title:"Quick start",href:"/"},{title:"Backend families",href:"/docs/introduction-backend-families"}]},{title:"Text classification",links:[{title:"Zero-shot text classification",href:"/docs/zero-shot-text-classification"},{title:"Few-shot text classification",href:"/docs/few-shot-text-classification"},{title:"Dynamic few-shot text classification",href:"/docs/dynamic-few-shot-text-classification"},{title:"Chain-of-thought text classification",href:"/docs/chain-of-thought-text-classification"},{title:"Tunable text classification",href:"/docs/tunable-text-classification"}]},{title:"Text-to-text modelling",links:[{title:"Text summarization",href:"/docs/text-summarization"},{title:"Text translation",href:"/docs/text-translation"},{title:"Tunable text-to-text",href:"/docs/tunable-text-to-text"}]},{title:"Text Vectorization",links:[{title:"Overview",href:"/docs/text-vectorization"}]},{title:"Tagging",links:[{title:"Overview",href:"/docs/tagging-overview"},{title:"Named Entity Recognition",href:"/docs/ner"}]},{title:"Contributing",links:[{title:"How to contribute",href:"/docs/how-to-contribute"}]}]}},function(e){e.O(0,[231,55,971,23,744],function(){return e(e.s=8356)}),_N_E=e.O()}]);