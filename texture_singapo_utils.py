from openai import OpenAI
import base64
from dotenv import load_dotenv
from rembg import remove
load_dotenv()

client = OpenAI()

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def get_image_description(image_path):
    """
    Get a description of the image using OpenAI's GPT-4o model.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        str: Description of the image.
    """
    # Encode the image
    base64_image = encode_image(image_path)

    # Create a chat completion request
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Provide a brief description of an object. Example response: A wooden cabinet with metal handles."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=250,
    )

    return response.choices[0].message.content

def get_object_mask(image_path,output_path):
    """
    Get the object mask from the image using rembg library.
    
    Args:
        image_path (str): Path to the image file.
        output_path (str): Path to save the output image.
    """

    with open(image_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input,only_mask=True)
            o.write(output)
        


