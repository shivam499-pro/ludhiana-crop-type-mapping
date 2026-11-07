// Project P7 - Rebuilt Script 04: Random Forest classification.
//
// Data provenance:
// - Training and held-out validation tables must be authorized Earth Engine
//   assets configured below. This script does not resample, split, or use
//   sample geometries.
// - The class labels are phenology-derived. Validation measures agreement
//   with those labels, not accuracy against independent field observations.
// - Feature calculations, AOI, dates, and QA60 masking follow Script 03.
//
// This draft only defines export tasks. It does not start them automatically;
// review and run this script in the Earth Engine Code Editor before exporting.

var AOI_ASSET = 'FAO/GAUL/2015/level2';
// Replace the project placeholder with an Earth Engine project you can access.
// Upload or copy authorized tables under these asset names before running.
var TRAINING_ASSET =
  'projects/YOUR_EARTH_ENGINE_PROJECT/assets/P7_training_points';
var VALIDATION_ASSET =
  'projects/YOUR_EARTH_ENGINE_PROJECT/assets/P7_validation_points';
var EXPORT_FOLDER = 'P7_CropMap_Ludhiana';

var CLASS_PROPERTY = 'class';
var CLASS_VALUES = [1, 2, 3];

// Keep this order aligned with the verified training and validation tables.
var PREDICTORS = [
  'B11_kharif',
  'B4_kharif',
  'B8_kharif',
  'ndvi_delta',
  'ndvi_kharif_mean',
  'ndvi_kharif_peak',
  'ndvi_rabi_mean',
  'ndvi_rabi_peak'
];

var EXPECTED_TRAINING_COUNTS = {
  total: 608,
  classes: {'1': 202, '2': 201, '3': 205}
};
var EXPECTED_VALIDATION_COUNTS = {
  total: 292,
  classes: {'1': 98, '2': 99, '3': 95}
};

var aoi = ee.FeatureCollection(AOI_ASSET)
  .filter(ee.Filter.and(
    ee.Filter.eq('ADM1_NAME', 'Punjab'),
    ee.Filter.eq('ADM2_NAME', 'Ludhiana')
  ));
var training = ee.FeatureCollection(TRAINING_ASSET);
var validation = ee.FeatureCollection(VALIDATION_ASSET);

function maskS2Clouds(image) {
  var qa = image.select('QA60');
  var mask = qa.bitwiseAnd(1 << 10).eq(0)
    .and(qa.bitwiseAnd(1 << 11).eq(0));
  return image.updateMask(mask).divide(10000)
    .copyProperties(image, ['system:time_start']);
}

function addNDVI(image) {
  return image.addBands(
    image.normalizedDifference(['B8', 'B4']).rename('NDVI')
  );
}

// Reproduce Script 03's collection filters and monthly median NDVI features.
// filterDate's end date is exclusive; the existing Script 03 date bounds are
// intentionally retained here rather than silently changed.
var S2 = 'COPERNICUS/S2_SR_HARMONIZED';
var s2_kharif = ee.ImageCollection(S2)
  .filterBounds(aoi)
  .filterDate('2024-06-01', '2024-10-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
  .map(maskS2Clouds)
  .map(addNDVI);

var s2_rabi = ee.ImageCollection(S2)
  .filterBounds(aoi)
  .filterDate('2024-11-01', '2025-04-30')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
  .map(maskS2Clouds)
  .map(addNDVI);

var kharif_months = ee.List.sequence(6, 10);
var kharif_monthly = ee.ImageCollection(kharif_months.map(function(m) {
  var month = ee.Number(m);
  return s2_kharif
    .filter(ee.Filter.calendarRange(month, month, 'month'))
    .select('NDVI')
    .median()
    .clip(aoi)
    .set('system:time_start', ee.Date.fromYMD(2024, month, 15).millis());
}));

var nov = s2_rabi.filterDate('2024-11-01', '2024-11-30')
  .select('NDVI')
  .median()
  .clip(aoi)
  .set('system:time_start', ee.Date.fromYMD(2024, 11, 15).millis());
var dec = s2_rabi.filterDate('2024-12-01', '2024-12-31')
  .select('NDVI')
  .median()
  .clip(aoi)
  .set('system:time_start', ee.Date.fromYMD(2024, 12, 15).millis());
var rabi_monthly_2025 = ee.ImageCollection(
  ee.List.sequence(1, 4).map(function(m) {
    var month = ee.Number(m);
    var start = ee.Date.fromYMD(2025, month, 1);
    return s2_rabi
      .filterDate(start, start.advance(1, 'month'))
      .select('NDVI')
      .median()
      .clip(aoi)
      .set('system:time_start', ee.Date.fromYMD(2025, month, 15).millis());
  })
);
var rabi_monthly = ee.ImageCollection([nov, dec]).merge(rabi_monthly_2025);

var ndvi_kharif_peak = kharif_monthly.select('NDVI').max()
  .clip(aoi)
  .rename('ndvi_kharif_peak');
var ndvi_kharif_mean = kharif_monthly.select('NDVI').mean()
  .clip(aoi)
  .rename('ndvi_kharif_mean');
var ndvi_rabi_peak = rabi_monthly.select('NDVI').max()
  .clip(aoi)
  .rename('ndvi_rabi_peak');
var ndvi_rabi_mean = rabi_monthly.select('NDVI').mean()
  .clip(aoi)
  .rename('ndvi_rabi_mean');

// Script 03 uses the Kharif seasonal median composite for B4, B8, and B11.
// No explicit B11 resampling is added here.
var s2_kharif_composite = ee.ImageCollection(S2)
  .filterBounds(aoi)
  .filterDate('2024-06-01', '2024-10-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
  .map(maskS2Clouds)
  .median()
  .clip(aoi);

var b4_kharif = s2_kharif_composite.select('B4').rename('B4_kharif');
var b8_kharif = s2_kharif_composite.select('B8').rename('B8_kharif');
var b11_kharif = s2_kharif_composite.select('B11').rename('B11_kharif');
var ndvi_delta = ndvi_kharif_peak
  .subtract(ndvi_rabi_peak)
  .rename('ndvi_delta');

// Build the same eight features as Script 03 in its original band order.
var featureStack = ndvi_kharif_peak
  .addBands(ndvi_rabi_peak)
  .addBands(ndvi_kharif_mean)
  .addBands(ndvi_rabi_mean)
  .addBands(b4_kharif)
  .addBands(b8_kharif)
  .addBands(b11_kharif)
  .addBands(ndvi_delta)
  .clip(aoi);

print('P7 feature-band projections', {
  ndvi_kharif_peak: featureStack.select('ndvi_kharif_peak').projection(),
  ndvi_rabi_peak: featureStack.select('ndvi_rabi_peak').projection(),
  ndvi_kharif_mean: featureStack.select('ndvi_kharif_mean').projection(),
  ndvi_rabi_mean: featureStack.select('ndvi_rabi_mean').projection(),
  B4_kharif: featureStack.select('B4_kharif').projection(),
  B8_kharif: featureStack.select('B8_kharif').projection(),
  B11_kharif: featureStack.select('B11_kharif').projection(),
  ndvi_delta: featureStack.select('ndvi_delta').projection()
});


Map.centerObject(aoi, 10);
Map.addLayer(aoi, {color: '1A6B3A', fillColor: '00000000', width: 2},
  'Ludhiana Boundary');
Map.addLayer(featureStack.select('ndvi_kharif_peak'),
  {min: 0, max: 1, palette: ['#d73027', '#ffffbf', '#1a9850']},
  'Kharif Peak NDVI', false);

var NUMERIC_TYPES = ee.List(['Integer', 'Long', 'Float', 'Double', 'Number']);

function countInvalidRows(collection) {
  var checked = collection.map(function(feature) {
    var numericPredictors = ee.List(PREDICTORS).map(function(name) {
      return NUMERIC_TYPES.contains(
        ee.Algorithms.ObjectType(feature.get(ee.String(name)))
      );
    }).contains(false);
    var numericClass = NUMERIC_TYPES.contains(
      ee.Algorithms.ObjectType(feature.get(CLASS_PROPERTY))
    );
    var knownClass = ee.List(CLASS_VALUES).contains(
      feature.get(CLASS_PROPERTY)
    );
    var invalidRecord = ee.Number(
      ee.Algorithms.If(numericPredictors, 1, 0)
      ).add(
      ee.Number(ee.Algorithms.If(numericClass, 0, 1))
      ).add(
      ee.Number(ee.Algorithms.If(knownClass, 0, 1))
      ).gt(0);
    return feature.set('_p7_invalid_record', invalidRecord);
  });
  return checked.filter(ee.Filter.eq('_p7_invalid_record', true)).size();
}

var audit = ee.Dictionary({
  aoiFeatureCount: aoi.size(),
  trainingCount: training.size(),
  trainingClassCounts: training.aggregate_histogram(CLASS_PROPERTY),
  trainingInvalidRows: countInvalidRows(training),
  trainingProperties: ee.Feature(training.first()).propertyNames(),
  validationCount: validation.size(),
  validationClassCounts: validation.aggregate_histogram(CLASS_PROPERTY),
  validationInvalidRows: countInvalidRows(validation),
  validationProperties: ee.Feature(validation.first()).propertyNames()
});

function matchesExpectedCounts(actual, expected) {
  if (!actual || Object.keys(actual).length !== CLASS_VALUES.length) {
    return false;
  }
  return CLASS_VALUES.every(function(classValue) {
    var key = String(classValue);
    return Object.prototype.hasOwnProperty.call(actual, key) &&
      actual[key] === expected[key];
  });
}

function hasRequiredProperties(actualProperties) {
  return actualProperties &&
    PREDICTORS.concat([CLASS_PROPERTY]).every(function(name) {
      return actualProperties.indexOf(name) !== -1;
    });
}

audit.evaluate(function(result, error) {
  if (error) {
    print('P7 preflight failed. No model or export tasks were created:', error);
    return;
  }

  print('P7 input audit:', result);
  var failures = [];

  if (result.aoiFeatureCount !== 1) {
    failures.push('Expected exactly one Ludhiana AOI feature.');
  }
  if (result.trainingCount !== EXPECTED_TRAINING_COUNTS.total) {
    failures.push('Unexpected training record count.');
  }
  if (!matchesExpectedCounts(
    result.trainingClassCounts,
    EXPECTED_TRAINING_COUNTS.classes
  )) {
    failures.push('Training class values or counts do not match the verified data.');
  }
  if (result.trainingInvalidRows !== 0) {
    failures.push('Training table has missing, non-numeric, or unknown values.');
  }
  if (!hasRequiredProperties(result.trainingProperties)) {
    failures.push('Training table is missing required feature/class properties.');
  }
  if (result.validationCount !== EXPECTED_VALIDATION_COUNTS.total) {
    failures.push('Unexpected validation record count.');
  }
  if (!matchesExpectedCounts(
    result.validationClassCounts,
    EXPECTED_VALIDATION_COUNTS.classes
  )) {
    failures.push('Validation class values or counts do not match the verified data.');
  }
  if (result.validationInvalidRows !== 0) {
    failures.push('Validation table has missing, non-numeric, or unknown values.');
  }
  if (!hasRequiredProperties(result.validationProperties)) {
    failures.push('Validation table is missing required feature/class properties.');
  }

  if (failures.length > 0) {
    print('P7 preflight failed. No model or export tasks were created:', failures);
    return;
  }

  print('Training and validation tables passed count, class, and numeric-type checks.');
  print('Training table only; held-out validation table is not used for fitting.');
  print('Predictor order:', PREDICTORS);
  print('Validation labels are phenology-derived, not independent field observations.');

  var classifier = ee.Classifier.smileRandomForest({
    numberOfTrees: 50,
    seed: 42
  }).train({
    features: training,
    classProperty: CLASS_PROPERTY,
    inputProperties: PREDICTORS
  });

  var classified = featureStack.classify(classifier)
    .clip(aoi)
    .rename('classification')
    .toInt8();
  Map.addLayer(classified, {
    min: 1,
    max: 3,
    palette: ['#1a9850', '#fee08b', '#d73027']
  }, 'Rebuilt Random Forest Classification');

  var validationPredictions = validation.classify(
    classifier,
    'predicted_class'
  );
  var confusionMatrix = validationPredictions.errorMatrix(
    CLASS_PROPERTY,
    'predicted_class',
    CLASS_VALUES
  );
  print('Held-out validation confusion matrix (rows=true, columns=predicted):',
    confusionMatrix);
  print('Held-out validation overall accuracy:', confusionMatrix.accuracy());
  print('Held-out validation Kappa:', confusionMatrix.kappa());
  print('Held-out validation producers accuracy by class:',
    confusionMatrix.producersAccuracy());
  print('Held-out validation consumers accuracy by class:',
    confusionMatrix.consumersAccuracy());

  // Emit a property-only table; the source CSV geometries are empty.
  var validationExport = validationPredictions.map(function(feature) {
    var properties = ee.Dictionary({
      sample_id: feature.get('system:index'),
      true_class: feature.get(CLASS_PROPERTY),
      predicted_class: feature.get('predicted_class')
    }).combine(feature.toDictionary(PREDICTORS), true);
    return ee.Feature(null, properties);
  });

  Export.image.toDrive({
    image: classified,
    description: 'P7_S04_RF_Classification_REBUILT_10m_EPSG32643',
    fileNamePrefix: 'P7_final_classified_map_rebuilt',
    folder: EXPORT_FOLDER,
    region: aoi.geometry(),
    // TODO: Reconcile this 10 m export with Script 03's 30 m sample scale
    // and Sentinel-2 B11's native 20 m resolution before changing resolution.
    scale: 10,
    crs: 'EPSG:32643',
    maxPixels: 1e13
  });

  Export.table.toDrive({
    collection: validationExport,
    description: 'P7_S04_RF_Validation_Predictions_REBUILT',
    fileNamePrefix: 'P7_validation_predictions_rebuilt',
    folder: EXPORT_FOLDER,
    fileFormat: 'CSV',
    selectors: ['sample_id', 'true_class', 'predicted_class']
      .concat(PREDICTORS)
  });
});
