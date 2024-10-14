text = """<|begin_of_text|>You are an expert at writing business rule code from the instruction given to you. Below is the instruction given write the most accurate business rule code.\n\n### Instruction:\nWrite a business rule that checks if the attributes "skeffective" and "skApplications" have changed. If yes, it triggers the processing of the "efficacyDocument" entity for the graph. If not, the rule returns FALSE.\n\n### Input:\n\n\n### Response:\nIIF[HaveOnlyChangedAttributes["efficacyDocument"], SendEntityForGraphProcessing["efficacyDocument"], FALSE]<|end_of_text|>"""


print(text[int(text.index("Response"))+len("Response"):])

