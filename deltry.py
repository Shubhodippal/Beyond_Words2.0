import streamlit as st
import os

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        st.success(f"File '{file_path}' has been deleted.")
    else:
        st.error(f"File '{file_path}' does not exist.")

def main():
    st.title("File Deletion App")

    # Text input for file name
    file_name = ('output.txt')
    
    if file_name:
        if st.button("Delete File"):
            delete_file(file_name)

if __name__ == "__main__":
    main()
