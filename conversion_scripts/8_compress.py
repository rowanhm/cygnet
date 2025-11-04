from lxml import etree

print('Compressing...')

# Parse the input XML file
tree = etree.parse('cygnet.xml')
root = tree.getroot()

# Remove all Provenance elements
for provenance in root.xpath('.//Provenance'):
    provenance.getparent().remove(provenance)

# Write the modified XML to output file with pretty print
tree.write('cygnet_small.xml', encoding='utf-8', xml_declaration=True, pretty_print=True)

print("Done! Provenance elements removed.")