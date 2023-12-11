'''
Assignment No: 05
Name: Yash Pramod Dhage
Roll: 70
Batch: B4
Title: Implement regular expression function to find URL, IP address, Date, PAN number in textual data using python libraries

'''

import re
def find_entities(text):

    result = {
        'URLs': re.findall(r'https?://\S+|www\.\S+', text),
        'IP Addresses': re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text),
        'Dates': re.findall(r'([1-9]|[12][0-9]|3[01])\/(0[1-9]|1[1,2])\/(19|20)\d{2}', text),
        'PAN Numbers': re.findall(r'[A-Z]{5}[0-9]{4}[A-Z]', text),
    }
    return result

# Example usage:
sample_text = """
First Dataset
Visit our website at https://www.google.com.
For support, contact us at support@example.com.
IP address: 192.168.0.1
Date: 11/22/2023
PAN number: AUJPD4551B

Second Dataset
Visit our website at https://www.youtube.com.
For more info connect with  info@example.com.
IP address: 192.148.7.1
Date: 30/10/2023
PAN number: CYRKD1290J
"""

result = find_entities(sample_text)

for entity_type, entities in result.items():
    print(f"{entity_type}: {entities}")



"""
Output:

URLs: ['https://www.google.com.', 'https://www.youtube.com.']
IP Addresses: ['192.168.0.1', '192.148.7.1']
Dates: ['11/22/2023', '30/10/2023']
PAN Numbers: ['AUJPD4551B', 'CYRKD1290J']

"""