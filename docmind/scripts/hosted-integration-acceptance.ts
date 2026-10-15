// One-shot DocMind hosted integration test. Runs real application operations.
// The test document is synthetic, isolated by a random marker, and deleted in finally.
// Auth comes from a Railway reference variable; credentials are never logged.
const origin = "https://aiengginer-production.up.railway.app";
const apiKey = Bun.env.DOCMIND_API_KEY;
if (!apiKey || apiKey.length < 24) throw new Error("DOCMIND_ACCEPTANCE_KEY_MISSING");
const marker = Array.from(crypto.getRandomValues(new Uint8Array(8))).map(n => n.toString(16).padStart(2, "0")).join("");
const filename = "acceptance-" + marker + ".txt";
const fixture = "DocMind acceptance record " + marker + ". The correct policy owner for this unique document is the quality assurance team. No customer or production data is included.";
let documentId: string | null = null;
const checks: Record<string, string> = {};
function assert(condition: unknown, code: string): asserts condition {
  if (!condition) throw new Error(code);
}
async function call(path: string, init: RequestInit = {}): Promise<{status:number, body:any}> {
  const headers = new Headers(init.headers);
  headers.set("X-API-Key", apiKey);
  headers.set("Accept", "application/json");
  const response = await fetch(origin + path, {...init, headers, redirect:"manual", signal:AbortSignal.timeout(120000)});
  assert(![301,302,303,307,308].includes(response.status), "UNEXPECTED_REDIRECT");
  const raw = await response.text();
  assert(raw.length <= 100000, "RESPONSE_TOO_LARGE");
  let body: any;
  try { body = JSON.parse(raw); } catch { body = {error:"NON_JSON"}; }
  return {status:response.status, body};
}
let failure: unknown = null;
try {
  const ready = await call("/ready");
  assert(ready.status === 200 && ready.body.database === "ok" && ready.body.vector_store === "ok",
    "DATABASE_OR_VECTOR_NOT_READY");
  checks["durable_dependencies"] = "PASS";
  const data = new FormData();
  data.set("file", new File([fixture], filename, {type:"text/plain"}));
  const uploaded = await call("/api/v1/documents/upload", {method:"POST",body:data});
  assert(uploaded.status === 200 && uploaded.body.status === "indexed"
      && typeof uploaded.body.id === "string" && uploaded.body.chunk_count >= 1, 
      "REAL_UPLOAD_OR_INDEXING_FAILED_" + uploaded.status);
  documentId = uploaded.body.id;
  checks["real_upload_and_vector_indexing"] = "PASS";
  const loaded = await call("/api/v1/documents/" + encodeURIComponent(documentId));
  assert(loaded.status === 200 && loaded.body.id === documentId, "DOCUMENT_NOT_PERSISTED");
  checks["real_document_readback"] = "PASS";
  const queried = await call("/api/v1/query", {
    method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({query:"Who is the policy owner for the acceptance record "+marker+"?",
                         filters:{doc_id:documentId},top_k:3}),
  });
  assert(queried.status === 200 && typeof queried.body.answer_status === "string" 
    && Number.isFinite(queried.body.retrieval_latency_ms), "REAL_QUERY_FAILED_" + queried.status);
  const answered = queried.body.answer_status === "answered"
     && Array.isArray(queried.body.citations) && queried.body.citations.length > 0
     && queried.body.citations.every((c:any)=>c.doc_id === documentId);
  checks["real_retrieval_request"] = "PASS";
  checks["llm_answer_with_verified_citations"] = answered ? "PASS" : "NOT_VERIFIED";
} catch (error) {
  failure = error;
} finally {
  if (documentId) {
    try {
      const deleted = await call("/api/v1/documents/"+encodeURIComponent(documentId),{method:"DELETE"});
      assert(deleted.status === 200 && deleted.body.success === true,"DOCUMENT_DELETE_FAILED");
      const absent = await call("/api/v1/documents/"+encodeURIComponent(documentId));
      assert(absent.status === 404,"DOCUMENT_STILL_VISIBLE_AFTER_DELETE");
      checks["real_document_cleanup"] = "PASS";
    } catch(error) {
      checks["real_document_cleanup"] = "FAILED";
      failure = failure ?? error;
    }
  }
}
console.log("DOCMIND_REAL_HOSTED_RESULT " + JSON.stringify({checks,origin,
    fixture:"synthetic_unique_and_cleaned",
    scope:"Live HTTPS upload/index/query/delete. LLM is NOT accepted unless a citation-grounded answer passes."}));
if (failure) throw failure;
await Bun.sleep(1500); // flush reporting before service exits
