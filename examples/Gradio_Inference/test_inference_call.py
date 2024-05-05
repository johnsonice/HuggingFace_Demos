from gradio_client import Client

client = Client("https://537326731225e437cd.gradio.live/")
result = client.predict(
		name="Hello!!",
		api_name="/predict"
)
print(result)