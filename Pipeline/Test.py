import requests

url = "http://localhost:8001/predict"

with open(r"C:\Study\Major\Spoon\Tests\Test3.jpeg", "rb") as image_file:
    response = requests.post(
        url,
        files={
            "image": (
                "Test3.jpeg",
                image_file,
                "image/jpeg",
            )
        },
        timeout=300,
    )

print(response.status_code)
print(response.json())