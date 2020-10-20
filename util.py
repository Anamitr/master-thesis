def save_to_file(output_file_name:str, content:str):
    output_file = open(output_file_name, "w")
    output_file.write(content)
    output_file.close()
