from lxml import etree

def open_xml(file_name):
    return etree.parse(file_name)


def _parser_size(root_element):
    size_data = root_element.xpath('size')[0]
    size_props = {}
    for element in size_data.getchildren():
        size_props[element.tag] = element.text
    return size_props


def _parser_objects(root_element):
    objects_list = []
    objects = root_element.xpath('object')
    for obj in objects:
        object_props = {
            'name': obj.find('name').text
        }
        bndbox = obj.xpath('bndbox')[0]
        for dim in bndbox.getchildren():
            object_props[dim.tag] = int(dim.text)
        objects_list.append(object_props)
    return objects_list


def parser_xml(root_element):
    size_props = _parser_size(root_element)
    objects_data = _parser_objects(root_element)
    return size_props, objects_data




