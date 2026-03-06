import{c as f,a as l}from"./bmn8XlW7.js";import"./EMQoTJgY.js";import{f as p}from"./DpkolLiq.js";import{I as c,s as m}from"./DxBGAPM1.js";import{l as u,s as $}from"./TXeUdaq1.js";function y(t,o){const r=u(o,["children","$$slots","$$events","$$legacy"]);/**
 * @license lucide-svelte v0.469.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const a=[["polygon",{points:"6 3 20 12 6 21 6 3"}]];c(t,$({name:"play"},()=>r,{get iconNode(){return a},children:(e,s)=>{var n=f(),i=p(n);m(i,o,"default",{}),l(e,n)},$$slots:{default:!0}}))}function _(t,o=2){if(t===0)return"0 Bytes";const r=1024,a=o<0?0:o,e=["Bytes","KB","MB","GB","TB"],s=Math.floor(Math.log(t)/Math.log(r));return parseFloat((t/Math.pow(r,s)).toFixed(a))+" "+e[s]}function x(t){if(t<60)return`${Math.floor(t)}s`;const o=Math.floor(t/3600),r=Math.floor(t%3600/60);return o>0?`${o}h ${r}m`:`${r}m`}function F(t){return t>=1e6?(t/1e6).toFixed(1)+"M":t>=1e3?(t/1e3).toFixed(1)+"K":t.toString()}export{y as P,_ as a,F as b,x as f};
