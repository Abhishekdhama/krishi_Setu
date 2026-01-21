from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse


app = Flask(__name__)


@app.route("/webhook", methods=['POST'])
def webhook():
    
    incoming_msg = request.values.get('Body', '').lower()
    print(f"Received a message: {incoming_msg}")

    
    resp = MessagingResponse()
    msg = resp.message()
    
    
    msg.body(f"Hi! Your message '{incoming_msg}' was received. The bot is connected!")

    
    return str(resp)

if __name__ == "__main__":
    app.run(debug=True)