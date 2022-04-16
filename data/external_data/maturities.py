from datetime import datetime

import numpy as np


def convert_maturity(maturity: str, today: str = None, round: int = 3) -> float:
    """Convert the given maturity, formatted as "%Y-%m-%d", into the the number of day between now (or "today") and this maturity, normalized as 1 = 365.

    Args:
        maturity (str): A str date formatted as "%Y-%m-%d"
        today (str, optional): If not provided, today will be computed as the today datetime object.

    Returns:
        float: The normalized (1 = 365) number of day(s) between maturity and today.
    """
    # Conver the given str maturity into a datetime object
    maturity_datetime = datetime.strptime(maturity, "%Y-%m-%d")

    if today:
        # Convert the provided str date into a datetime object
        today_datetime = datetime.strptime(today, "%Y-%m-%d")
    else:
        # If today is not provided, use the real date to build the corresponding datetime object
        today_datetime = datetime.today()

    # Return it as a number of day when one year is one
    converted_maturity = (maturity_datetime - today_datetime).days/365
    if converted_maturity > 0:
        return np.round(converted_maturity, 3)
    else:
        raise ValueError("Maturity is less than one day from now.")
