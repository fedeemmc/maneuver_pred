import xml.etree.ElementTree as ET
from datetime import datetime

def parse_time(t):
    return datetime.strptime(t, "%Y/%m/%dT%H:%M:%S.%f")

def get_text(parent, tag, default=None):
    node = parent.find(tag)
    if node is None or node.text is None:
        return default
    return node.text

def read_stations(xml_file):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    stations = []

    for event in root.find("Events").findall("Event"):

        station = {
            "id": get_text(event, "eventID"),
            "aos": parse_time(get_text(event, "start")),
            "los": parse_time(get_text(event, "stop")),
            "duration": get_text(event, "duration"),
            "el_max": float(get_text(event, "elMax", 0.0)),
        }

        stations.append(station)

    return stations