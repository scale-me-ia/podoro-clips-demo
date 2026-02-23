/**
 * scan-youtube — Podoro Edge Function
 * =====================================
 * Creates a scan_request row in Supabase. The local cron (scan_youtube.py)
 * polls this table and processes pending requests.
 *
 * POST /functions/v1/scan-youtube
 * Body (optional):
 *   { "podcast_id": "uuid", "days_back": 7, "trigger_clips": true }
 *
 * Response:
 *   { "request_id": "uuid", "status": "queued", "message": "..." }
 */

import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL") ?? "";
const SUPABASE_SERVICE_ROLE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Content-Type": "application/json",
};

serve(async (req: Request) => {
  // Handle CORS preflight
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  if (req.method !== "POST") {
    return new Response(
      JSON.stringify({ error: "Method not allowed. Use POST." }),
      { status: 405, headers: corsHeaders }
    );
  }

  try {
    const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE);

    // Parse optional body
    let body: Record<string, unknown> = {};
    try {
      body = await req.json();
    } catch {
      // empty body is fine
    }

    const podcast_id = body.podcast_id as string | undefined;
    const days_back = (body.days_back as number) ?? 7;
    const trigger_clips = (body.trigger_clips as boolean) ?? false;

    // Validate days_back
    if (days_back < 1 || days_back > 90) {
      return new Response(
        JSON.stringify({ error: "days_back must be between 1 and 90" }),
        { status: 400, headers: corsHeaders }
      );
    }

    // Check for a very recent pending request (debounce: 5 min)
    const fiveMinAgo = new Date(Date.now() - 5 * 60 * 1000).toISOString();
    const { data: recent } = await supabase
      .from("scan_requests")
      .select("id, created_at, status")
      .eq("status", "pending")
      .gte("created_at", fiveMinAgo)
      .maybeSingle();

    if (recent) {
      return new Response(
        JSON.stringify({
          message: "A scan was already requested recently. Please wait.",
          request_id: recent.id,
          status: "already_queued",
          queued_at: recent.created_at,
        }),
        { status: 200, headers: corsHeaders }
      );
    }

    // Insert scan request
    const { data, error } = await supabase
      .from("scan_requests")
      .insert({
        status: "pending",
        podcast_id: podcast_id ?? null,
        days_back,
        trigger_clips,
        triggered_by: "api",
      })
      .select("id, created_at")
      .single();

    if (error) {
      console.error("Failed to insert scan_request:", error);
      return new Response(
        JSON.stringify({ error: "Failed to queue scan", details: error.message }),
        { status: 500, headers: corsHeaders }
      );
    }

    return new Response(
      JSON.stringify({
        message: "YouTube scan queued. The local worker will process it within 6 hours, or on next cron run.",
        request_id: data.id,
        status: "queued",
        queued_at: data.created_at,
        params: { podcast_id, days_back, trigger_clips },
      }),
      { status: 202, headers: corsHeaders }
    );
  } catch (err) {
    console.error("scan-youtube error:", err);
    return new Response(
      JSON.stringify({ error: "Internal server error", details: String(err) }),
      { status: 500, headers: corsHeaders }
    );
  }
});
