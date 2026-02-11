# API Reference

## Authentication

### Register
`POST /auth/register`

Register a new user account.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securepassword"
}
```

**Response:**
```json
{
  "access_token": "eyJhbG...",
  "token_type": "bearer",
  "user": {
    "id": "user_123",
    "email": "user@example.com",
    "created_at": "2024-01-01T00:00:00"
  }
}
```

### Login
`POST /auth/login`

Authenticate a user.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securepassword"
}
```

**Response:**
Same as Register.

---

## Documents

### Upload Document
`POST /documents/upload`

Upload a PDF document for analysis.

**Form Data:**
- `file`: (Binary PDF file)
- `ticker`: (String, e.g., "AAPL")

**Response:**
```json
{
  "document_id": "doc_123",
  "num_chunks": 15,
  "num_pages": 5,
  "s3_key": "user_id/filename.pdf",
  "status": "success"
}
```

### List Documents
`GET /documents/`

List all uploaded documents for the current user.

**Response:**
```json
[
  {
    "document_id": "doc_123",
    "ticker": "AAPL",
    "filename": "report.pdf",
    "created_at": "..."
  }
]
```

### Get Document
`GET /documents/{document_id}`

Get metadata for a specific document.

---

## Analysis

### Analyze Document
`POST /analysis/{document_id}`

Run the analysis workflow on a document.

**Request Body:**
```json
{
  "query": "What is the revenue growth?"
}
```

**Response:**
```json
{
  "answer": "The revenue grew by 5%...",
  "verification_status": "PASS",
  "intelligence_hub": {
    "key_highlights": [...],
    "risk_assessment": {...},
    "sentiment": {...}
  },
  "metadata": {...}
}
```
