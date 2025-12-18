from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Path
from typing import List, Optional
from langchain.retrievers import ParentDocumentRetriever
from app.core.dependencies import get_parent_document_retriever
from app.middleware.auth_s2s import verify_internal_api_key
from app.services.data_service import DataService

router = APIRouter(prefix="/admin/knowledge")


# API Import Excel
@router.post("/import-excel")
async def import_excel(
    file: UploadFile = File(...),
    file_id: str = Form(...),
    retriever: ParentDocumentRetriever = Depends(get_parent_document_retriever),
    authorized: bool = Depends(verify_internal_api_key),
):
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="File must be Excel format")
        
    try:
        DataService.ingest_excel(file, file_id, retriever)
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
    file_id: str = Form(...),
    topic: str = Form(...),
    location: str = Form(...),
    name: Optional[str] = Form(None),
    source: Optional[str] = Form(None),
    retriever: ParentDocumentRetriever = Depends(get_parent_document_retriever),
    authorized: bool = Depends(verify_internal_api_key)
):
    SUPPORTED_FILE_TYPES = ['application/pdf',
                            'application/msword',
                            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                            'text/plain']

    # Gom metadata từ Form vào dict
    metadata = {
        "Topic": topic,
        "Location": location,
        "Name": name if name else file.filename,
        "Source": source if source else "Unknown"
    }

    if file.content_type not in SUPPORTED_FILE_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    if topic.strip() == "" or location.strip() == "":
        raise HTTPException(status_code=400, detail="Topic and Location are required fields")

    try:
        DataService.ingest_unstructured_file(file, file_id, metadata, retriever)
        return {
            "status": "success",
            "message": f"File '{file.filename}' imported successfully."
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


# API Delete
@router.delete("/{file_id}")
async def delete_knowledge(
    file_id: str = Path(...),
    retriever: ParentDocumentRetriever = Depends(get_parent_document_retriever),
    authorized: bool = Depends(verify_internal_api_key),
):
    try:
        DataService.delete_document(file_id, retriever)
        return {
            "status": "success",
            "message": f"Document {file_id} has been deleted."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


# API Update
@router.put("/{file_id}")
async def update_knowledge_file(
    file_id: str = Path(...),
    file: UploadFile = File(...),
    topic: str = Form(...),
    location: str = Form(...),
    name: Optional[str] = Form(None),
    source: Optional[str] = Form(None),
    retriever: ParentDocumentRetriever = Depends(get_parent_document_retriever),
    authorized: bool = Depends(verify_internal_api_key),
):
    try:
        # Xóa cái cũ
        DataService.delete_document(file_id, retriever)
        
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