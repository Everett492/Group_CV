with open('Result_3_17.txt', 'r') as f:
    f2 = open('Res_retrained.txt', 'w')
    for data in f:
        data = data.split()
        if data[0] == 'Test':
            f2.write(f"Epoch {data[2]}: {data[4]}\n")
    f2.close()
