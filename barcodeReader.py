from dbr import DynamsoftBarcodeReader
dbr = DynamsoftBarcodeReader()
dbr.initLicense('LICENSE-KEY')
 
params = dbr.getParameters()
print(params)
 
import json
json_obj = json.loads(params)
# Update JSON object
templateName = json_obj['ImageParameter']['Name']
# DPM
json_obj['ImageParameter']['DPMCodeReadingModes'][0]['Mode'] = 'DPMCRM_GENERAL'
json_obj['ImageParameter']['LocalizationModes'][0]['Mode'] = 'LM_STATISTICS_MARKS'
# Convert JSON object to string
params = json.dumps(json_obj)
# Set parameters
ret = dbr.setParameters(params)
 
results = dbr.decodeFile('dpm.jpg', dbr.BF_ALL)
for result in results:
    print('barcode format: ' + result[0])
    print('barcode value: ' + result[1])