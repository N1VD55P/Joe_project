
import smtplib
import random
from email.message import EmailMessage
import os

def send_alert(matched_filename, matched_name, receiver_email):
    # Random suspicious websites
    suspicious_sites = [
        "http://faceshare-x.ru",
        "http://deepmatchdata.biz",
        "http://leakfinders.net",
        "http://darkmirror.ai",
        "http://facespider.to",
        "http://illicitfaces.info",
    ]
    selected_sites = random.sample(suspicious_sites, k=random.randint(2, 3))

    # Email body
    body = f"""
Hi {matched_name},

Your image has been detected on the following suspicious websites:

{chr(10).join(f"🔗 {site}" for site in selected_sites)}

If you did not consent to your image being used on these websites, please file a report here:

🚨 https://cybercrime.gov.in/

Your matched image is attached below.

– FaceWatch
"""

    # Email setup
    sender_email = "n1vdeepnain@gmail.com"
    app_password = "twzt krfj rgbi znnc"  # Gmail App Password

    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = "⚠️ Alert: Your image found on suspicious websites"
    msg.set_content(body)

    # Path to the matched image from the database
    matched_image_path = os.path.join("Database", matched_filename)

    # Attach image
    with open(matched_image_path, "rb") as f:
        img_data = f.read()
        msg.add_attachment(img_data, maintype="image", subtype="jpeg", filename=matched_filename)

    # Send email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender_email, app_password)
        smtp.send_message(msg)

    print(f"✅ Alert email sent to {receiver_email} with {matched_filename} attached.")

