import pandas as pd

def airport_k_anonymization(data, k=3):
    sensitive_attrs = ['Passenger ID', 'Departure Time', 'Arrival Time']
    
    # Anonymize sensitive attributes
    for attr in sensitive_attrs:
        if attr in data.columns:
            data[attr] = data[attr].apply(lambda x: '***' + x[-4:] if isinstance(x, str) else x)
    
    # Sorting based on sensitive attributes
    sorted_data = data.sort_values(by=sensitive_attrs)
    
    # Divide data into groups
    groups = [sorted_data.iloc[i:i+k] for i in range(0, len(sorted_data), k)]
    
    # Checking for k-anonymity
    for group in groups:
        values_count = group[sensitive_attrs].apply(pd.value_counts).fillna(0).min(axis=1)
        min_count = values_count.min()
        if min_count < k:
            print('The group does not satisfy {}-anonymity requirement.'.format(k))
            break
    
    # Concatenate groups
    anonymized_data = pd.concat(groups, ignore_index=True)
    return anonymized_data

if __name__ == '__main__':
    # Sample airport data
    airport_data = pd.DataFrame({
        'Passenger Name': ['John Doe', 'Jane Smith', 'Michael Johnson', 'Emily Brown', 'Robert Wilson', 'Sophia Lee'],
        'Passenger ID': ['ABC1234', 'DEF5678', 'GHI9101', 'JKL1121', 'MNO3141', 'PQR5162'],
        'Flight Number': ['AA123', 'UA456', 'DL789', 'AA123', 'UA456', 'DL789'],
        'Departure Time': ['2024-03-29 08:00', '2024-03-29 09:30', '2024-03-29 12:15', '2024-03-29 08:00', '2024-03-29 09:30', '2024-03-29 12:15'],
        'Arrival Time': ['2024-03-29 10:00', '2024-03-29 11:45', '2024-03-29 14:30', '2024-03-29 10:00', '2024-03-29 11:45', '2024-03-29 14:30'],
        'Seat Number': ['12A', '25B', '5C', '14D', '30E', '7F']
    })

    anon_airport_data = airport_k_anonymization(airport_data, k=3)

    # Outputting results
    print(anon_airport_data.to_string(index=False))
