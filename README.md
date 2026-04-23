# fastapi-rag-mercedes

endpoints 
Post api/ai/ask 
        {
            brand : str
            model : str
            year: int
            chassis_code: str
        }

Post 
create new manual 
api/manuals/
    {
        file:""
        file_path:""
        brand : str
        model : str
        year: int
        chassis_code: str
        created_at
    }

Put
api/manuals/{title}
    {
        file:""
        file_path:""
        brand : str
        model : str
        year: int
        chassis_code: str
        updated_at
    }
