import sys

from src.logger import logging

def error_message_detail(error, error_detial:sys):
    _,_,exc_tb= error_detial.exc_info()

    file_name=exc_tb.tb_frame.f_code.co_filename

    error_message="Error Occured in Python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
            
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detial:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detial=error_detial)


    def __str__(self) -> str:
        return self.error_message
    

# if __name__=="__main__":
#     try:
#         a=1/0
#     except Exception as e:
#         logging.info("There is devided by Zero Error")
#         raise CustomException(e,sys)