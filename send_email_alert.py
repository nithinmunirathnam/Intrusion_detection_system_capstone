# import smtplib
# import os
# from email.message import EmailMessage
#
# def send_intruder_email(image_path):
#     # Email details
#     to_email = "munirathnamkpm21@gmail.com"
#     from_email = "munirathnamkpm21@gmail.com"
#     app_password = "mynqrgetamvsemsp"
#
#     # Create the email
#     msg = EmailMessage()
#     msg['Subject'] = '🚨 Intruder Alert at Home!'
#     msg['From'] = from_email
#     msg['To'] = to_email
#     msg.set_content('An intruder was detected. See the attached image.')
#
#     # Attach the image
#     with open(image_path, 'rb') as f:
#         img_data = f.read()
#         msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=os.path.basename(image_path))
#
#     # Send the email
#     try:
#         with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
#             smtp.login(from_email, app_password)
#             smtp.send_message(msg)
#         print("✅ Email sent successfully!")
#     except Exception as e:
#         print("❌ Failed to send email:", e)
#
# # Actual image path
# image_path = r"C:\Users\NITHIN\Desktop\intrusion_detection\intruders\intruder_2025-04-08_18-26-26.jpg"
# send_intruder_email(image_path)
import smtplib
import os
from email.message import EmailMessage
from pathlib import Path


def send_intruder_email(folder_path):
    # Get the latest image file in the folder
    image_files = sorted(Path(folder_path).glob("*.jpg"), key=os.path.getmtime, reverse=True)

    if not image_files:
        print("❌ No image files found in the folder.")
        return

    image_path = image_files[0]  # Latest file

    # Email credentials and settings
    to_email = "munirathnamkpm21@gmail.com"
    from_email = "munirathnamkpm21@gmail.com"
    app_password = "mynqrgetamvsemsp"

    # Create email message
    msg = EmailMessage()
    msg['Subject'] = '🚨 Intruder Alert at Home!'
    msg['From'] = from_email
    msg['To'] = to_email
    msg.set_content('An intruder was detected. See the attached image.')

    # Attach the latest image
    with open(image_path, 'rb') as f:
        img_data = f.read()
        msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=os.path.basename(image_path))

    # Send email
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(from_email, app_password)
            smtp.send_message(msg)
        print(f"✅ Email sent with image: {image_path}")
    except Exception as e:
        print("❌ Failed to send email:", e)


# Folder path containing intruder images
folder_path = r"C:\Users\NITHIN\Desktop\intrusion_detection\intruders"
send_intruder_email(folder_path)
