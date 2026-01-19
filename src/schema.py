import pandera as pa
from pandera import Column, Check, DataFrameSchema

# We use the functional API (DataFrameSchema) instead of the Class API
# This is more compatible with older/different versions of the library.

AutoMPGSchema = DataFrameSchema({
    "Cylinders": Column(int, checks=[Check.ge(3), Check.le(12)], coerce=True),
    "Displacement": Column(float, checks=Check.ge(0), coerce=True),
    "Horsepower": Column(float, checks=Check.ge(0), coerce=True),
    "Weight": Column(float, checks=Check.ge(0), coerce=True),
    "Acceleration": Column(float, checks=Check.ge(0), coerce=True),
    # Note: We map "Model Year" strictly.
    "Model Year": Column(int, checks=[Check.ge(70), Check.le(82)], coerce=True),
    "Origin": Column(int, checks=[Check.ge(1), Check.le(3)], coerce=True),

    # Target variable (nullable because we might run inference on new data)
    "MPG": Column(float, checks=Check.ge(0), nullable=True, required=False, coerce=True),
})
