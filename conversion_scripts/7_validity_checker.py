#!/usr/bin/env python3
"""
Simple XML Schema validator
"""
from lxml import etree
import sys


def validate_xml(xml_file, xsd_file):
    """
    Validate an XML file against an XSD schema.

    Args:
        xml_file: Path to the XML file
        xsd_file: Path to the XSD schema file

    Returns:
        True if valid, False otherwise
    """
    try:
        print('Parse the XSD schema')
        with open(xsd_file, 'r', encoding='utf-8') as f:
            schema_root = etree.XML(f.read().encode('utf-8'))
        schema = etree.XMLSchema(schema_root)

        print('Parse the XML document')
        with open(xml_file, 'r', encoding='utf-8') as f:
            xml_doc = etree.parse(f)

        print('Validating')
        is_valid = schema.validate(xml_doc)

        if is_valid:
            print(f"✓ {xml_file} is VALID according to {xsd_file}")
            return True
        else:
            print(f"✗ {xml_file} is INVALID according to {xsd_file}")
            print("\nValidation errors:")
            for error in schema.error_log:
                print(f"  Line {error.line}: {error.message}")
            return False

    except etree.XMLSchemaParseError as e:
        print(f"✗ Error parsing XSD schema: {e}")
        return False
    except etree.XMLSyntaxError as e:
        print(f"✗ Error parsing XML document: {e}")
        return False
    except FileNotFoundError as e:
        print(f"✗ File not found: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


if __name__ == "__main__":

    xml_file = sys.argv[1] if len(sys.argv) > 1 else 'cygnet.xml'
    xsd_file = sys.argv[2] if len(sys.argv) > 2 else 'cygnet.xsd'

    result = validate_xml(xml_file, xsd_file)
    sys.exit(0 if result else 1)