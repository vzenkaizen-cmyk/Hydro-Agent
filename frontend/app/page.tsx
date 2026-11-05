 "use client";
import { useEffect, useMemo, useState } from "react";
import { Activity, Waves, CloudRain, Factory, LayoutDashboard, BrainCircuit, CalendarDays, RefreshCw, Download, Droplets, Zap, BarChart3 } from "lucide-react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, BarChart, Bar } from "recharts";

const API = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");
const horizons = [7,14,30,90,180,365], validationWindows = [7,14,30,60,90];
type AnyRow = Record<string, any>;
function fmt(v:any, digits=0){ if(v===null||v===undefined||!Number.isFinite(Number(v))) return "N/A"; return Number(v).toLocaleString("en-US",{maximumFractionDigits:digits,minimumFractionDigits:digits});}
function dateLabel(v:string){ if(!v) return "—"; const d=new Date(v+"T00:00:00"); return d.toLocaleDateString("en-GB",{day:"2-digit",month:"short",year:"numeric"});}
function downloadCsv(rows:AnyRow[], filename:string){ if(!rows?.length)return; const cols=Object.keys(rows[0]); const csv=[cols.join(","),...rows.map(r=>cols.map(c=>{const s=String(r[c]??"");return `"${s.replace(/"/g,'""')}"`}).join(","))].join("\n"); const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([csv],{type:"text/csv;charset=utf-8;"}));a.download=filename;a.click();URL.revokeObjectURL(a.href);}
function Metric({label,value,sub,icon:Icon}:any){return <div className="metric"><div style={{display:"flex",justifyContent:"space-between",alignItems:"center"}}><div className="metric-label">{label}</div><Icon size={17} color="#8ba0b1"/></div><div className="metric-value">{value}</div><div className="metric-sub">{sub}</div></div>}
function Table({rows,columns}: {rows:AnyRow[],columns:{key:string,label:string,digits?:number,date?:boolean}[]}){return <div className="table-wrap"><table><thead><tr>{columns.map(c=><th key={c.key}>{c.label}</th>)}</tr></thead><tbody>{rows?.length?rows.map((r,i)=><tr key={i}>{columns.map(c=><td key={c.key}>{c.date?dateLabel(r[c.key]):(typeof r[c.key]==="number"?fmt(r[c.key],c.digits??1):(r[c.key]??"—"))}</td>)}</tr>):<tr><td colSpan={columns.length}>No records available.</td></tr>}</tbody></table></div>}
export default function Home(){
 const [site,setSite]=useState("BBO"),[forecastDays,setForecastDays]=useState(30),[validationDays,setValidationDays]=useState(30);
 const [data,setData]=useState<any>(null),[loading,setLoading]=useState(false),[error,setError]=useState(""),[tab,setTab]=useState("daily");
 const [live,setLive]=useState<any>(null),[liveLoading,setLiveLoading]=useState(false),[liveError,setLiveError]=useState("");
 const [sites,setSites]=useState<string[]>([]);
 async function runLiveAgent(){setLiveLoading(true);setLiveError("");setLive(null);try{const r=await fetch(`${API}/api/live-agent?site=${encodeURIComponent(site)}`,{cache:"no-store"});const j=await r.json();if(!r.ok)throw new Error(j.detail||"Live weather agent failed");setLive(j);}catch(e:any){setLiveError(e.message||"Could not run the live weather agent");}finally{setLiveLoading(false);}}
 async function load(){setLoading(true);setError("");try{const r=await fetch(`${API}/api/dashboard?site=${encodeURIComponent(site)}&forecast_days=${forecastDays}&validation_days=${validationDays}`,{cache:"no-store"});const j=await r.json();if(!r.ok)throw new Error(j.detail||"Unable to load dashboard");setData(j);}catch(e:any){setError(e.message||"API connection failed. Check the backend URL.");}finally{setLoading(false);}}
 useEffect(()=>{fetch(`${API}/api/plants`).then(r=>r.json()).then(j=>{if(j.sites?.length){setSites(j.sites);if(!j.sites.includes(site))setSite(j.sites[0]);}}).catch(()=>{});},[]);
 useEffect(()=>{load();},[site,forecastDays,validationDays]);
 const forecastChart=useMemo(()=>data?.forecast?.map((r:AnyRow)=>({...r,day:dateLabel(r.Date)}))||[],[data]);
 const testChart=useMemo(()=>data?.test_results?.map((r:AnyRow)=>({...r,day:dateLabel(r.Date)}))||[],[data]);
 const next=data?.next_forecast;
 return <div className="shell">
  <aside className="sidebar">
   <div className="brand"><div className="brand-icon"><Waves size={22}/></div><div><div className="brand-name">Hydro Intelligence</div><div className="brand-sub">Forecasting Agent</div></div></div>
   <div className="nav-label">Workspace</div><div className="nav-item active"><LayoutDashboard size={16}/> Overview</div><div className="nav-item"><BarChart3 size={16}/> Generation Forecast</div><div className="nav-item"><CloudRain size={16}/> Rainfall Analysis</div><div className="nav-item"><BrainCircuit size={16}/> Model Validation</div>
   <div className="nav-label">System</div><div className="nav-item"><Factory size={16}/> Plant Performance</div><div className="nav-item"><CalendarDays size={16}/> Forecast Reports</div>
   <div className="sidebar-foot">Hydro Intelligence Agent<br/>Plant-specific XGBoost forecasting<br/>Vidullanka PLC</div>
  </aside>
  <main className="main">
   <div className="topbar"><div><div className="eyebrow">Hydro generation planning</div><h1 className="page-title">Hydro Intelligence Agent</h1></div><div className="status"><Activity size={13}/> Forecasting service</div></div>
   <section className="hero"><div className="hero-kicker">Hydro generation planning</div><div className="hero-title">Forecast with confidence. Plan with intelligence.</div><p className="hero-copy">Hydropower generation forecasting using historical plant performance, rainfall patterns and plant-specific XGBoost models.</p></section>
   <section className="controls">
    <div className="field"><label>Select Hydro Plant</label><select value={site} onChange={e=>setSite(e.target.value)}>{(sites.length?sites:["BBO","BTO","BKN","EME"]).map(s=><option key={s}>{s}</option>)}</select></div>
    <div className="field"><label>Forecast horizon</label><select value={forecastDays} onChange={e=>setForecastDays(Number(e.target.value))}>{horizons.map(n=><option value={n} key={n}>{n} days</option>)}</select></div>
    <div className="field"><label>Historical validation window</label><select value={validationDays} onChange={e=>setValidationDays(Number(e.target.value))}>{validationWindows.map(n=><option value={n} key={n}>{n} days</option>)}</select></div>
    <button className="refresh" onClick={load} disabled={loading}><RefreshCw size={13} style={{verticalAlign:"middle",marginRight:6}}/>{loading?"Updating…":"Refresh forecast"}</button><button className="download" onClick={runLiveAgent} disabled={liveLoading}>{liveLoading?"Running live agent…":"Run live weather agent"}</button>
   </section>
   {loading&&!data?<div className="loading">Preparing plant-specific XGBoost model and generating forecasts…</div>:null}
   {error?<div className="error"><b>Unable to load forecasting data.</b><br/>{error}<br/><small>Confirm that the Python API is running and NEXT_PUBLIC_API_BASE_URL is configured.</small></div>:null}
   {data&&<>
    <div className="section-head"><div><h2 className="section-title">{site} Generation Forecast</h2><div className="section-note">Latest recorded actual: {dateLabel(data.last_actual_date)} · Next forecast: {dateLabel(next?.Date)}</div></div><button className="download" onClick={()=>downloadCsv(data.future_comparison,`${site}_future_forecast.csv`)}><Download size={13}/> Download forecast</button></div>
    <div className="grid">
     <Metric label="Forecast Generation" value={`${fmt(next?.Forecast_Generation_kWh)} kWh`} sub="Tomorrow's predicted generation" icon={Zap}/>
     <Metric label="Assumed Rainfall" value={`${fmt(next?.Forecast_Rainfall_mm,1)} mm`} sub="Historical same-month median" icon={CloudRain}/>
     <Metric label="Latest Actual Generation" value={`${fmt(data.latest_actual_generation)} kWh`} sub="Most recent recorded generation" icon={Factory}/>
     <Metric label="Latest Actual Date" value={dateLabel(data.last_actual_date)} sub="Latest date in historical dataset" icon={CalendarDays}/>
    </div>
    <div className="note warn">{data.rainfall_note} The latest actual and tomorrow's forecast refer to different dates; their difference is not a forecast error.</div>
    {liveError?<div className="error" style={{marginTop:12}}>Live weather agent: {liveError}. Confirm OPENWEATHER_API_KEY is configured on the Python API host and that this is a hydro plant supported by the agent.</div>:null}
    {live?<><div className="section-head"><div><h2 className="section-title">Live OpenWeather Agent · {live.plant?.name||site}</h2><div className="section-note">Short-range weather forecast combined with the existing XGBoost generation model</div></div><button className="download" onClick={()=>downloadCsv(live.forecast,`${site}_live_weather_agent_forecast.csv`)}><Download size={13}/> Download live forecast</button></div><Table rows={live.forecast} columns={[{key:"Date",label:"Date",date:true},{key:"Forecast_Rainfall_mm",label:"Forecast rainfall (mm)",digits:2},{key:"Rain_Probability",label:"Rain probability",digits:2},{key:"Forecast_Generation_kWh",label:"Forecast generation (kWh)"},{key:"Forecast_Source",label:"Forecast source"}]}/></>:null}
    <div className="section-head"><div><h2 className="section-title">Upcoming {forecastDays}-Day Forecast</h2><div className="section-note">Daily predicted generation and rainfall scenario for {site}</div></div></div>
    <div className="chart-grid">
     <div className="card"><h3 className="chart-title">Forecast Generation (kWh)</h3><div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><LineChart data={forecastChart}><CartesianGrid strokeDasharray="3 3" stroke="#edf1f5"/><XAxis dataKey="day" tick={{fontSize:9}} minTickGap={24}/><YAxis tick={{fontSize:10}}/><Tooltip/><Line type="monotone" dataKey="Forecast_Generation_kWh" name="Forecast generation" stroke="#1677b8" strokeWidth={2.5} dot={false}/></LineChart></ResponsiveContainer></div></div>
     <div className="card"><h3 className="chart-title">Forecast Rainfall (mm)</h3><div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><BarChart data={forecastChart}><CartesianGrid strokeDasharray="3 3" stroke="#edf1f5"/><XAxis dataKey="day" tick={{fontSize:9}} minTickGap={24}/><YAxis tick={{fontSize:10}}/><Tooltip/><Bar dataKey="Forecast_Rainfall_mm" name="Assumed rainfall" fill="#16866c" radius={[3,3,0,0]}/></BarChart></ResponsiveContainer></div></div>
    </div>
    <div className="section-head"><h2 className="section-title">Upcoming Daily Generation and Rainfall</h2></div>
    <Table rows={data.future_comparison} columns={[{key:"Date",label:"Date",date:true},{key:"Forecast_Generation_kWh",label:"Forecast generation (kWh)"},{key:"Forecast_Rainfall_mm",label:"Assumed rainfall (mm)",digits:2},{key:"Actual_Generation_kWh",label:"Actual generation (kWh)"},{key:"Actual_Rainfall_mm",label:"Actual rainfall (mm)",digits:2},{key:"Error_Percentage",label:"Error (%)",digits:2}]}/>
    <div className="section-head"><h2 className="section-title">Upcoming Monthly Generation and Rainfall</h2></div>
    <Table rows={data.future_monthly} columns={[{key:"Month",label:"Month"},{key:"Forecast_Generation_kWh",label:"Forecast generation (kWh)"},{key:"Forecast_Rainfall_mm",label:"Forecast rainfall scenario (mm)",digits:1},{key:"Days",label:"Days"}]}/>
    <div className="section-head"><div><h2 className="section-title">Historical Actual vs Forecast Comparison</h2><div className="section-note">Chronological split: {fmt(data.counts.train)} training · {fmt(data.counts.validation)} validation · {fmt(data.counts.test)} untouched test rows</div></div><button className="download" onClick={()=>downloadCsv(data.test_results,`${site}_historical_validation.csv`)}><Download size={13}/> Download validation</button></div>
    <div className="grid">
     <Metric label="Test MAE" value={`${fmt(data.metrics.MAE)} kWh`} sub="Mean absolute error" icon={Activity}/>
     <Metric label="Test RMSE" value={`${fmt(data.metrics.RMSE)} kWh`} sub="Root mean squared error" icon={Activity}/>
     <Metric label="Test MAPE" value={data.metrics.MAPE===null?"N/A":`${fmt(data.metrics.MAPE,2)}%`} sub="Mean absolute percentage error" icon={BarChart3}/>
     <Metric label="Test R²" value={fmt(data.metrics.R2,3)} sub="Coefficient of determination" icon={BrainCircuit}/>
    </div>
    <div className="note">Best validation MAE: {fmt(data.metrics.Validation_MAE)} kWh. This is a chronological one-day-ahead holdout evaluation. Measured historical rainfall is supplied as an input for each test day; future forecasts use historical same-month median rainfall assumptions.</div>
    <div className="card" style={{marginTop:15}}>
     <div className="tabs"><button className={`tab ${tab==="daily"?"active":""}`} onClick={()=>setTab("daily")}>Daily Comparison</button><button className={`tab ${tab==="monthly"?"active":""}`} onClick={()=>setTab("monthly")}>Monthly Comparison</button></div>
     {tab==="daily"?<>
      <h3 className="chart-title">Daily Actual vs Forecast Generation</h3><div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><LineChart data={testChart}><CartesianGrid strokeDasharray="3 3" stroke="#edf1f5"/><XAxis dataKey="day" tick={{fontSize:9}} minTickGap={25}/><YAxis tick={{fontSize:10}}/><Tooltip/><Legend/><Line dataKey="Actual_Generation_kWh" name="Actual" stroke="#16866c" strokeWidth={2} dot={false}/><Line dataKey="Forecast_Generation_kWh" name="Forecast" stroke="#1677b8" strokeWidth={2} dot={false}/></LineChart></ResponsiveContainer></div>
      <div style={{height:15}}/><h3 className="chart-title">Daily Measured Rainfall Used by the Model</h3><div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><BarChart data={testChart}><CartesianGrid strokeDasharray="3 3" stroke="#edf1f5"/><XAxis dataKey="day" tick={{fontSize:9}} minTickGap={25}/><YAxis tick={{fontSize:10}}/><Tooltip/><Bar dataKey="Actual_Rainfall_mm" name="Measured rainfall" fill="#16866c"/></BarChart></ResponsiveContainer></div>
      <div style={{height:15}}/><Table rows={data.test_results} columns={[{key:"Date",label:"Date",date:true},{key:"Actual_Generation_kWh",label:"Actual generation"},{key:"Forecast_Generation_kWh",label:"Forecast generation"},{key:"Generation_Difference_kWh",label:"Difference"},{key:"Absolute_Error_kWh",label:"Absolute error"},{key:"Error_Percentage",label:"Error (%)",digits:2},{key:"Actual_Rainfall_mm",label:"Measured rainfall",digits:2}]}/>
     </>:<>
      <h3 className="chart-title">Monthly Actual vs Forecast Generation</h3><div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><BarChart data={data.monthly_test}><CartesianGrid strokeDasharray="3 3" stroke="#edf1f5"/><XAxis dataKey="Month" tick={{fontSize:10}}/><YAxis tick={{fontSize:10}}/><Tooltip/><Legend/><Bar dataKey="Actual_Generation_kWh" name="Actual" fill="#16866c"/><Bar dataKey="Forecast_Generation_kWh" name="Forecast" fill="#1677b8"/></BarChart></ResponsiveContainer></div>
      <div style={{height:15}}/><Table rows={data.monthly_test} columns={[{key:"Month",label:"Month"},{key:"Days_Compared",label:"Days compared"},{key:"Actual_Generation_kWh",label:"Actual generation"},{key:"Forecast_Generation_kWh",label:"Forecast generation"},{key:"Generation_Difference_kWh",label:"Difference"},{key:"Error_Percentage",label:"Error (%)",digits:2}]}/>
     </>}
    </div>
   </>}
   <div className="footer">Hydro Intelligence Agent · Existing plant-specific XGBoost forecasting workflow · Vidullanka PLC</div>
  </main>
 </div>;
}
