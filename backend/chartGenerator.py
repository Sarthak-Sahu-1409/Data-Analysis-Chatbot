import matplotlib.pyplot as plt  # Import the matplotlib pyplot module for plotting graphs
import io  # Import io module to handle in-memory binary streams
import base64  # Import base64 module to encode binary data to base64 string

def plot_and_encode(fig):
    # Create an in-memory bytes buffer to save the plot image
    buf = io.BytesIO()
    # Save the figure to the buffer in PNG format with tight bounding box to minimize whitespace
    fig.savefig(buf, format="png", bbox_inches='tight')
    # Move the buffer's cursor to the beginning so it can be read from the start
    buf.seek(0)
    # Read the image data from the buffer, encode it to base64, and decode to UTF-8 string
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    # Close the figure to free up memory
    plt.close(fig)
    # Return the base64 encoded string representation of the plot image
    return img_base64