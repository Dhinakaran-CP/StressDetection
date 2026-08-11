var D=__DATA_JSON__,F=__FEAT_JSON__,FC=__FC_JSON__,DS=__DS_JSON__,SBD=__SBD_JSON__,TBD=__TBD_JSON__;
var fD=D.slice(),fF=F.slice();
var LOADED=typeof Plotly!=="undefined";
var PL={paper_bgcolor:"#1f2937",plot_bgcolor:"#111827",font:{color:"#e5e7eb",size:11},margin:{t:40,r:20,b:50,l:50},xaxis:{gridcolor:"#374151"},yaxis:{gridcolor:"#374151"},legend:{bgcolor:"rgba(0,0,0,0)"}};

function sp(id,traces,layout){
  if(!LOADED){document.getElementById(id).innerHTML='<div class="chart-err">Plotly not loaded</div>';return;}
  try{Plotly.newPlot(id,traces,Object.assign({},PL,layout||{}),{responsive:true});}
  catch(e){console.error("Plot "+id+":",e);document.getElementById(id).innerHTML='<div class="chart-err">Error: '+e.message+'</div>';}
}

function selectDataset(ds){document.getElementById("fDS").value=ds;onDS();}

function onDS(){
  try{
    var ds=document.getElementById("fDS").value;
    var ss=document.getElementById("fSub"),ts=document.getElementById("fTask");
    ss.innerHTML='<option value="all">All</option>';
    ts.innerHTML='<option value="all">All</option>';
    if(ds!=="all"){
      (SBD[ds]||[]).forEach(function(s){ss.innerHTML+='<option value="'+s+'">'+s+'</option>';});
      (TBD[ds]||[]).forEach(function(t){ts.innerHTML+='<option value="'+t+'">'+t+'</option>';});
    }else{
      DS.forEach(function(d){
        (SBD[d]||[]).forEach(function(s){
          if(!Array.from(ss.options).some(function(o){return o.value===s;}))
            ss.innerHTML+='<option value="'+s+'">'+s+'</option>';
        });
      });
    }
    apply();
  }catch(e){console.error("onDS:",e);}
}

function apply(){
  try{
    var ds=document.getElementById("fDS").value;
    var sub=document.getElementById("fSub").value;
    var tsk=document.getElementById("fTask").value;
    var lbl=document.getElementById("fLbl").value;
    fD=[];fF=[];
    for(var i=0;i<D.length;i++){
      var r=D[i];
      if(ds!=="all"&&r.dataset!==ds)continue;
      if(sub!=="all"&&r.subject!==sub)continue;
      if(tsk!=="all"&&r.task!==tsk)continue;
      if(lbl!=="all"&&r.label!==parseInt(lbl))continue;
      fD.push(r);
      fF.push(i<F.length?F[i]:{});
    }
    updateTbl();plotTE();plotBox();plotVio();plotMH();plotMB();plotSW();plotSS();plotFD();plotFC();plotQ();plotM();
  }catch(e){console.error("apply:",e);}
}

function doReset(){document.getElementById("fDS").value="all";onDS();}

function updateTbl(){
  document.getElementById("tb").innerHTML=fD.slice(0,500).map(function(r){
    return '<tr><td>'+r.dataset+'</td><td>'+r.subject+'</td><td>'+r.task+'</td><td>'+r.window_index+'</td>'+
      '<td style="color:'+(r.stress_label==="Stressed"?"#ef4444":"#22c55e")+'">'+r.stress_label+'</td>'+
      '<td>'+r.quality_score.toFixed(2)+'</td><td>'+(r.has_physio?"Y":"N")+'</td>'+
      '<td>'+(r.has_audio?"Y":"N")+'</td><td>'+(r.has_video?"Y":"N")+'</td></tr>';
  }).join('');
}

function plotTE(){
  var fi=FC.indexOf(document.getElementById("fFeat").value);
  if(fi<0)return;
  var bd={};
  fD.forEach(function(r,i){
    if(!bd[r.dataset])bd[r.dataset]={x:[],y:[],c:[],t:[]};
    bd[r.dataset].x.push(r.window_index);
    var v=fF[i]?fF[i][FC[fi]]:null;
    bd[r.dataset].y.push(v);
    bd[r.dataset].c.push(r.stress_label);
    bd[r.dataset].t.push(r.subject+"|"+r.task+"|"+r.stress_label);
  });
  var tr=[];
  var cl={Calm:"#22c55e",Stressed:"#ef4444",Unknown:"#6b7280"};
  Object.keys(bd).forEach(function(ds){
    var d=bd[ds];
    ["Calm","Stressed","Unknown"].forEach(function(l){
      var idx=[];
      for(var j=0;j<d.c.length;j++){if(d.c[j]===l)idx.push(j);}
      if(!idx.length)return;
      tr.push({
        x:idx.map(function(j){return d.x[j];}),
        y:idx.map(function(j){return d.y[j];}),
        text:idx.map(function(j){return d.t[j];}),
        mode:"markers",name:ds+"-"+l,
        marker:{color:cl[l],size:4,opacity:0.7},
        hovertemplate:"%{text}<br>%{y:.4f}<extra></extra>"
      });
    });
  });
  sp("pTE",tr,{title:"Window Evolution: "+FC[fi],xaxis:PL.xaxis,yaxis:{gridcolor:"#374151",title:FC[fi]},height:420});
}

function plotBox(){
  var fi=FC.indexOf(document.getElementById("fFeat").value);
  if(fi<0)return;
  var cv=[],sv=[];
  fD.forEach(function(r,i){
    var v=fF[i]?fF[i][FC[fi]]:null;
    if(v===null||v===undefined||isNaN(v))return;
    if(r.stress_label==="Calm")cv.push(v);
    else if(r.stress_label==="Stressed")sv.push(v);
  });
  sp("pBox",[
    {y:cv,type:"box",name:"Calm",marker:{color:"#22c55e"}},
    {y:sv,type:"box",name:"Stressed",marker:{color:"#ef4444"}}
  ],{title:"Box: "+FC[fi],yaxis:{gridcolor:"#374151",title:FC[fi]},height:360});
}

function plotVio(){
  var fi=FC.indexOf(document.getElementById("fFeat").value);
  if(fi<0)return;
  var cv=[],sv=[];
  fD.forEach(function(r,i){
    var v=fF[i]?fF[i][FC[fi]]:null;
    if(v===null||v===undefined||isNaN(v))return;
    if(r.stress_label==="Calm")cv.push(v);
    else if(r.stress_label==="Stressed")sv.push(v);
  });
  if(cv.length<2&&sv.length<2){sp("pViolin",[],{title:"Violin: "+FC[fi],height:360});return;}
  var traces=[];
  if(cv.length>=2)traces.push({y:cv,type:"violin",name:"Calm",marker:{color:"#22c55e"},meanline:{visible:true}});
  if(sv.length>=2)traces.push({y:sv,type:"violin",name:"Stressed",marker:{color:"#ef4444"},meanline:{visible:true}});
  sp("pViolin",traces,{title:"Violin: "+FC[fi],yaxis:{gridcolor:"#374151",title:FC[fi]},height:360,violinmode:"overlay"});
}

function plotMH(){
  var mods=["has_physio","has_audio","has_video"],mn=["Physio","Audio","Video"];
  var z=mods.map(function(m){return DS.map(function(d){
    var dd=fD.filter(function(r){return r.dataset===d;});
    return dd.filter(function(r){return r[m];}).length/Math.max(dd.length,1);
  });});
  sp("pModHeat",[{
    z:z,x:DS,y:mn,type:"heatmap",colorscale:[[0,"#1f2937"],[1,"#3b82f6"]],
    text:z.map(function(r){return r.map(function(v){return (v*100).toFixed(0)+"%";});}),
    texttemplate:"%{text}"
  }],{title:"Modality Coverage",height:360,yaxis:{gridcolor:"#374151",autorange:"reversed"}});
}

function plotMB(){
  var mods=["has_physio","has_audio","has_video"],mn=["Physio","Audio","Video"],cl=["#3b82f6","#eab308","#a855f7"];
  sp("pModBar",mods.map(function(m,i){return {
    x:DS,y:DS.map(function(d){
      var dd=fD.filter(function(r){return r.dataset===d&&r[m];});
      return dd.length?dd.reduce(function(s,r){return s+r.quality_score;},0)/dd.length:0;
    }),type:"bar",name:mn[i],marker:{color:cl[i]};
  };}),{title:"Quality by Modality",barmode:"group",height:360,yaxis:{gridcolor:"#374151",title:"Mean Quality",range:[0,1]}});
}

function plotSW(){
  var sub=[];fD.forEach(function(r){if(sub.indexOf(r.subject)<0)sub.push(r.subject);});sub=sub.slice(0,30);
  sp("pSubWin",[{x:sub,y:sub.map(function(s){return fD.filter(function(r){return r.subject===s;}).length;}),
    type:"bar",marker:{color:"#3b82f6"}
  }],{title:"Windows/Subject",height:360,xaxis:{gridcolor:"#374151",tickangle:-45},yaxis:{gridcolor:"#374151",title:"Count"}});
}

function plotSS(){
  var ld=fD.filter(function(r){return r.label>=0;});var sub=[];
  ld.forEach(function(r){if(sub.indexOf(r.subject)<0)sub.push(r.subject);});sub=sub.slice(0,30);
  sp("pSubStr",[
    {x:sub,y:sub.map(function(s){return ld.filter(function(r){return r.subject===s&&r.stress_label==="Calm";}).length;}),
     type:"bar",name:"Calm",marker:{color:"#22c55e"}},
    {x:sub,y:sub.map(function(s){return ld.filter(function(r){return r.subject===s&&r.stress_label==="Stressed";}).length;}),
     type:"bar",name:"Stressed",marker:{color:"#ef4444"}}
  ],{title:"Stress/Subject",barmode:"stack",height:360,xaxis:{gridcolor:"#374151",tickangle:-45}});
}

function plotFD(){
  var fi=FC.indexOf(document.getElementById("fFeat").value);if(fi<0)return;
  var cv=[],sv=[];
  fD.forEach(function(r,i){
    var v=fF[i]?fF[i][FC[fi]]:null;
    if(v===null||v===undefined||isNaN(v))return;
    if(r.stress_label==="Calm")cv.push(v);else if(r.stress_label==="Stressed")sv.push(v);
  });
  sp("pFeatDist",[
    {x:cv,type:"histogram",name:"Calm",marker:{color:"#22c55e"},opacity:0.6,nbinsx:30},
    {x:sv,type:"histogram",name:"Stressed",marker:{color:"#ef4444"},opacity:0.6,nbinsx:30}
  ],{title:"Dist: "+FC[fi],barmode:"overlay",height:360,xaxis:{gridcolor:"#374151",title:FC[fi]},yaxis:{gridcolor:"#374151",title:"Count"}});
}

function plotFC(){
  var fs=FC.slice(0,8),dt=fF.slice(0,300);
  if(dt.length<5||fs.length<2){sp("pFeatCorr",[],{title:"Correlation (need more data)",height:360});return;}
  var vl=fs.map(function(f){return dt.map(function(r){var v=r[f];return v!==null&&v!==undefined&&!isNaN(v)?v:0;});});
  var n=fs.length,cr=[];
  for(var i=0;i<n;i++){cr[i]=[];for(var j=0;j<n;j++)cr[i][j]=0;}
  for(var i=0;i<n;i++)for(var j=0;j<n;j++){
    var mi=0,mj=0;for(var k=0;k<vl[i].length;k++){mi+=vl[i][k];mj+=vl[j][k];}
    mi/=vl[i].length;mj/=vl[j].length;var num=0,di=0,dj=0;
    for(var k=0;k<vl[i].length;k++){num+=(vl[i][k]-mi)*(vl[j][k]-mj);di+=(vl[i][k]-mi)*(vl[i][k]-mi);dj+=(vl[j][k]-mj)*(vl[j][k]-mj);}
    cr[i][j]=di>0&&dj>0?num/Math.sqrt(di*dj):0;
  }
  sp("pFeatCorr",[{z:cr,x:fs.map(function(f){return f.substring(0,15);}),y:fs.map(function(f){return f.substring(0,15);}),
    type:"heatmap",colorscale:"RdBu",zmin:-1,zmax:1
  }],{title:"Feature Correlation",height:360});
}

function plotQ(){
  var bd={};fD.forEach(function(r){if(!bd[r.dataset])bd[r.dataset]=[];bd[r.dataset].push(r.quality_score);});
  sp("pQual",Object.keys(bd).map(function(d){return {x:bd[d],type:"histogram",name:d.substring(0,15),opacity:0.6,nbinsx:20};}),
  {title:"Quality Distribution",barmode:"overlay",height:360,xaxis:{gridcolor:"#374151",title:"Score",range:[0,1.1]}});
}

function plotM(){
  var fs=FC.slice(0,15);
  var z=DS.map(function(d){return fs.map(function(f){
    var dd=fD.filter(function(r){return r.dataset===d;});
    var miss=0;
    dd.forEach(function(r){var idx=D.indexOf(r);var ff=idx<F.length?F[idx]:null;
      if(!ff||ff[f]===null||ff[f]===undefined)miss++;});
    return dd.length?miss/dd.length:0;
  });});
  sp("pMiss",[{z:z,x:fs.map(function(f){return f.substring(0,18);}),y:DS,type:"heatmap",
    colorscale:[[0,"#22c55e"],[1,"#ef4444"]],
    text:z.map(function(r){return r.map(function(v){return (v*100).toFixed(0)+"%";});}),
    texttemplate:"%{text}"
  }],{title:"Missingness",height:360});
}

onDS();
