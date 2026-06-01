# API Reference

Base URL: `https://api.mcbeat.io/v1`

All requests require `Authorization: Bearer <token>` header except `/auth` endpoints.

---

## Authentication

### POST /auth/register
```json
// Request
{ "email": "user@example.com", "password": "..." }

// Response
{ "user_id": "uuid", "token": "jwt_token" }
```

### POST /auth/login
```json
// Request
{ "email": "user@example.com", "password": "..." }

// Response
{ "token": "jwt_token", "user": { "id": "uuid", "plan": "creator" } }
```

---

## Projects

### GET /projects
Returns all projects for authenticated user.

### POST /projects
```json
// Request
{ "title": "My Summer Trip", "config": { "export_format": "tiktok", "caption_style": "bold" } }

// Response
{ "project_id": "uuid", "upload_urls": { "music": "presigned_r2_url", "clips": "presigned_r2_url" } }
```

### GET /projects/:id
Returns project details including job status.

---

## Jobs

### POST /jobs
Trigger a render job for a project.
```json
// Request
{ "project_id": "uuid", "type": "full_render" }

// Response
{ "job_id": "uuid", "status": "queued", "estimated_seconds": 120 }
```

### GET /jobs/:id
Poll job status.
```json
// Response
{
  "job_id": "uuid",
  "status": "processing", // queued | processing | complete | failed
  "progress": 65,         // 0-100
  "output_url": null      // populated when complete
}
```

---

## WebSocket

Connect to `wss://api.mcbeat.io/v1/ws/:job_id` for real-time job progress updates.

```json
// Messages received
{ "type": "progress", "value": 45, "stage": "beat_detection" }
{ "type": "progress", "value": 80, "stage": "rendering" }
{ "type": "complete", "output_url": "https://r2.mcbeat.io/..." }
{ "type": "error", "message": "FFmpeg encoding failed" }
```
