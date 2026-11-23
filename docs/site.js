"use strict";
const escapeHTML = value => String(value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
let atlas = [];
const byId = id => document.getElementById(id);
function row(entry) {
  const tags = entry.labels === null ? '<span class="tag none">Not mapped beyond rank 3</span>' : entry.labels.length ? entry.labels.map(label => `<span class="tag">${escapeHTML(label.replaceAll("_", " "))}</span>`).join("") : '<span class="tag none">None</span>';
  const ev = entry.explained_variance === undefined ? "" : `<div class="ev">Cumulative explained variance: ${(100*entry.explained_variance).toFixed(1)}%<div class="ev-meter"><span style="width:${Math.max(0,Math.min(100,100*entry.explained_variance))}%"></span></div></div>`;
  let audio = "";
  if (entry.audio) {
    const a = entry.audio;
    audio = `<div class="audio"><audio controls preload="none" src="${escapeHTML(a.file)}" aria-label="Listen to a fit example for ${escapeHTML(entry.text)}"></audio></div><div class="source"><a href="${escapeHTML(a.source_url)}" target="_blank" rel="noopener">${escapeHTML(a.filename)}</a> · CC0 · coefficient +${Number(a.coefficient).toFixed(4)}</div>`;
  }
  return `<div class="description"><span class="rank">Rank ${entry.rank}</span><p class="caption">${escapeHTML(entry.text)}</p><div class="tags" title="${escapeHTML(entry.rationale || "")}">${tags}</div>${ev}${audio}</div>`;
}
function render() {
  const query = byId("search").value.trim().toLowerCase();
  const block = byId("block").value;
  const selected = byId("selected").checked;
  const heads = atlas.filter(h => (block === "all" || String(h.block) === block) && (!selected || h.summary) && (`${h.block}/${String(h.head).padStart(2,"0")} ${h.summary || ""} ${h.entries.map(e=>`${e.text} ${(e.labels||[]).join(" ")}`).join(" ")}`.toLowerCase().replaceAll("_"," ").includes(query.replaceAll("_"," "))));
  byId("count").textContent = `${heads.length} of 64 heads · final stage · 32 ranks per head`;
  byId("heads").innerHTML = heads.length ? heads.map(h=>`<article class="head" id="head-${h.block}-${h.head}"><div class="head-title"><h3>Block ${h.block} · Head ${String(h.head).padStart(2,"0")}</h3><a href="#head-${h.block}-${h.head}" aria-label="Link to block ${h.block} head ${h.head}">#${h.block}/${String(h.head).padStart(2,"0")}</a></div>${h.summary ? `<p class="summary">${escapeHTML(h.summary)}</p>` : ""}${h.entries.slice(0,3).map(row).join("")}<details><summary>View ranks 4–32</summary><ol start="4">${h.entries.slice(3).map(e=>`<li>${escapeHTML(e.text)}${e.explained_variance === undefined ? "" : `<span class="ev"> · ${(100*e.explained_variance).toFixed(1)}% cumulative EV</span>`}</li>`).join("")}</ol></details></article>`).join("") : '<p class="empty">No heads match these filters. Try a shorter description or reset the block filter.</p>';
}
for (const id of ["search","block","selected"]) byId(id).addEventListener(id === "search" ? "input" : "change", render);
// Avoid simultaneous playback when exploring several heads.
document.addEventListener("play", event => {if(event.target.tagName === "AUDIO")document.querySelectorAll("audio").forEach(a=>{if(a!==event.target)a.pause();});},true);
fetch("atlas.json").then(response=>{if(!response.ok)throw new Error("Atlas download failed");return response.json();}).then(data=>{atlas=data.heads;byId("loading").hidden=true;render();if(location.hash.startsWith("#head-"))document.querySelector(location.hash)?.scrollIntoView();}).catch(error=>{byId("loading").textContent="The atlas could not be loaded. Reload the page or download atlas.json using the link below.";console.error(error);});
