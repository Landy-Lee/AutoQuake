def degree_trans(part: str):
    """Transform degree-minute-second to decimal degrees."""
    if len(part) == 7:
        deg = int(part[:2])
        if part[2:4] == '  ':
            value = 0
        else:
            value = int(part[2:4]) / 60
        dig = value + int(part[5:]) / 3600
    else:
        deg = int(part[:3])
        if part[3:5] == '  ':
            value = 0
        else:
            value = int(part[3:5]) / 60
        dig = value + int(part[6:]) / 3600
    return deg + dig
