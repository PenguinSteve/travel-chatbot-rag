from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Path
from typing import List, Optional
from langchain.retrievers import ParentDocumentRetriever
from app.core.dependencies import get_parent_document_retriever
from app.middleware.auth_jwt import get_current_user_payload_strict
from app.services.data_service import DataService

router = APIRouter(prefix="/admin/knowledge")


# API Import Excel
@router.post("/import-excel")
async def import_excel(
    file: UploadFile = File(...),
    retriever: ParentDocumentRetriever = Depends(get_parent_document_retriever),
    user_payload: dict = Depends(get_current_user_payload_strict),
):
    if user_payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to perform this action")

    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="File must be Excel format")
        
    try:
        DataService.ingest_excel(file, retriever)
        return {
            "status": "success",
            "message": f"Successfully imported documents from Excel file."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


# API Import Unstructured file (PDF, Word, Txt)
@router.post("/import-file")
async def import_file(
    file: UploadFile = File(...),
    topic: str = Form(...),
    location: str = Form(...),
    name: Optional[str] = Form(None),
    source: Optional[str] = Form(None),
    retriever: ParentDocumentRetriever = Depends(get_parent_document_retriever),
    user_payload: dict = Depends(get_current_user_payload_strict),
):
    if user_payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to perform this action")

    # Gom metadata từ Form vào dict
    metadata = {
        "Topic": topic,
        "Location": location,
        "Name": name if name else file.filename,
        "Source": source if source else "Unknown"
    }

    try:
        DataService.ingest_unstructured_file(file, metadata, retriever)
        return {
            "status": "success",
            "message": f"File '{file.filename}' imported successfully."
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


# API Delete
@router.delete("/{doc_id}")
async def delete_knowledge(
    doc_id: str = Path(..., description="MongoDB _id of the parent document"),
    retriever: ParentDocumentRetriever = Depends(get_parent_document_retriever),
    user_payload: dict = Depends(get_current_user_payload_strict),
):
    if user_payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to perform this action")

    try:
        DataService.delete_document(doc_id, retriever)
        return {
            "status": "success",
            "message": f"Document {doc_id} has been deleted."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


# API Update
@router.put("/{doc_id}")
async def update_knowledge_file(
    doc_id: str,
    file: UploadFile = File(...),
    topic: str = Form(...),
    location: str = Form(...),
    name: Optional[str] = Form(None),
    source: Optional[str] = Form(None),
    retriever: ParentDocumentRetriever = Depends(get_parent_document_retriever),
    user_payload: dict = Depends(get_current_user_payload_strict),
):
    if user_payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to perform this action")

    try:
        # Xóa cái cũ
        DataService.delete_document(doc_id, retriever)
        
        # Thêm cái mới
        metadata = {
            "Topic": topic,
            "Location": location,
            "Name": name if name else file.filename,
            "Source": source if source else "Unknown"
        }
        DataService.ingest_unstructured_file(file, metadata, retriever)
        
        return {
            "status": "success",
            "message": "Document updated successfully.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")