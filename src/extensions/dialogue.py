"""Schema-driven missing-data clarification. No invented values or dosing."""
def missing_data_dialogue(required,provided):
    return [{'field':field,'question':f'What is {field.replace("_"," ")}? You can leave it unknown.'}
            for field in required if provided.get(field) is None]
