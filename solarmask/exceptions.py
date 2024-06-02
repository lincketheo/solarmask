class DataNotFoundException(Exception):
    """Exception raised when the required data does not exist."""

    def __init__(self, data_product_description: str):
        self.message = "Data product " + data_product_description + " does not exist."
        super().__init__(self.message)
