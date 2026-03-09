"""
Session memory management for ClinAssist
Tracks structured symptom state and prevents field overwrites
"""
from typing import Dict, Any, List, Optional, Set
import json
from config import REQUIRED_FIELDS
from database import (
    get_symptom_record,
    update_symptom_record,
    get_session_history,
    get_asked_fields,
    set_asked_fields,
)


class SessionMemory:
    """
    Manages session state and symptom data collection
    Prevents overwriting non-null fields and tracks missing attributes
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.symptom_data: Dict[str, Any] = {}
        self.asked_fields: List[str] = []
        self.conversation_history: List[Dict[str, str]] = []
        self.load_state()
    
    def load_state(self):
        """Load current state from database"""
        # Load symptom record
        record = get_symptom_record(self.session_id)
        if record:
            # Load basic fields
            for field in REQUIRED_FIELDS:
                value = record.get(field)
                if value is not None:
                    self.symptom_data[field] = value
            
            # Load risk and summary metadata
            for field in ["risk_level", "risk_reason", "recommended_action", "summary"]:
                value = record.get(field)
                if value is not None:
                    self.symptom_data[field] = value
        
        # Load conversation history
        self.conversation_history = get_session_history(self.session_id)

        # Load asked fields
        self.asked_fields = get_asked_fields(self.session_id)
    
    def update_fields(self, new_data: Dict[str, Any]):
        """
        Update symptom fields with no-overwrite policy
        Only updates fields that are currently None/empty
        
        Args:
            new_data: Dictionary with new field values from LLM extraction
        """
        updated = False
        
        for field in REQUIRED_FIELDS:
            if field in new_data:
                new_value = new_data[field]
                
                # Skip if new value is None or empty
                if new_value is None:
                    continue
                
                # Skip empty strings
                if isinstance(new_value, str) and not new_value.strip():
                    continue
                
                # Skip empty lists
                if isinstance(new_value, list) and len(new_value) == 0:
                    continue
                
                # Only update if current value is None or empty
                current_value = self.symptom_data.get(field)
                
                if current_value is None:
                    self.symptom_data[field] = new_value
                    updated = True
                elif isinstance(current_value, list) and len(current_value) == 0:
                    self.symptom_data[field] = new_value
                    updated = True
                elif isinstance(current_value, str) and not current_value.strip():
                    self.symptom_data[field] = new_value
                    updated = True
        
        # Persist to database if any updates
        if updated:
            self.persist()
    
    def get_missing_fields(self) -> List[str]:
        """
        Get list of fields that are still missing
        
        Returns:
            List of field names that are None or empty
        """
        missing = []
        
        for field in REQUIRED_FIELDS:
            value = self.symptom_data.get(field)
            
            if value is None:
                missing.append(field)
            elif isinstance(value, str) and not value.strip():
                missing.append(field)
            elif isinstance(value, list) and len(value) == 0:
                missing.append(field)
        
        return missing
    
    def is_intake_complete(self) -> bool:
        """
        Check if all 7 required fields are filled
        
        Returns:
            True if all fields have valid values, False otherwise
        """
        return len(self.get_missing_fields()) == 0
    
    def mark_field_asked(self, field_name: str):
        """Mark a field as having been asked about"""
        if field_name not in self.asked_fields:
            self.asked_fields.append(field_name)
            set_asked_fields(self.session_id, self.asked_fields)
    
    def get_unasked_missing_fields(self) -> List[str]:
        """Get missing fields that haven't been explicitly asked about yet"""
        missing = self.get_missing_fields()
        return [f for f in missing if f not in self.asked_fields]
    
    def persist(self):
        """Save current symptom data to database"""
        update_symptom_record(self.session_id, self.symptom_data)
    
    def get_symptom_data(self) -> Dict[str, Any]:
        """Get current symptom data"""
        return self.symptom_data.copy()
    
    def get_progress(self) -> Dict[str, bool]:
        """
        Get completion status for each field
        
        Returns:
            Dictionary mapping field names to completion boolean
        """
        progress = {}
        
        for field in REQUIRED_FIELDS:
            value = self.symptom_data.get(field)
            
            if value is None:
                progress[field] = False
            elif isinstance(value, str) and not value.strip():
                progress[field] = False
            elif isinstance(value, list) and len(value) == 0:
                progress[field] = False
            else:
                progress[field] = True
        
        return progress
