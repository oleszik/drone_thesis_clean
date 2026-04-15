export function isTencentProvider(mapProvider) {
  return String(mapProvider || "").toLowerCase() === "tencent";
}

function outOfChina(lng, lat) {
  return lng < 72.004 || lng > 137.8347 || lat < 0.8293 || lat > 55.8271;
}

function transformLat(x, y) {
  let ret = -100.0 + (2.0 * x) + (3.0 * y) + (0.2 * y * y) + (0.1 * x * y) + (0.2 * Math.sqrt(Math.abs(x)));
  ret += ((20.0 * Math.sin(6.0 * x * Math.PI)) + (20.0 * Math.sin(2.0 * x * Math.PI))) * (2.0 / 3.0);
  ret += ((20.0 * Math.sin(y * Math.PI)) + (40.0 * Math.sin((y / 3.0) * Math.PI))) * (2.0 / 3.0);
  ret += ((160.0 * Math.sin((y / 12.0) * Math.PI)) + (320 * Math.sin((y * Math.PI) / 30.0))) * (2.0 / 3.0);
  return ret;
}

function transformLng(x, y) {
  let ret = 300.0 + x + (2.0 * y) + (0.1 * x * x) + (0.1 * x * y) + (0.1 * Math.sqrt(Math.abs(x)));
  ret += ((20.0 * Math.sin(6.0 * x * Math.PI)) + (20.0 * Math.sin(2.0 * x * Math.PI))) * (2.0 / 3.0);
  ret += ((20.0 * Math.sin(x * Math.PI)) + (40.0 * Math.sin((x / 3.0) * Math.PI))) * (2.0 / 3.0);
  ret += ((150.0 * Math.sin((x / 12.0) * Math.PI)) + (300.0 * Math.sin((x / 30.0) * Math.PI))) * (2.0 / 3.0);
  return ret;
}

export function wgs84ToGcj02(lng, lat) {
  const lon = Number(lng);
  const latitude = Number(lat);
  if (!Number.isFinite(lon) || !Number.isFinite(latitude) || outOfChina(lon, latitude)) {
    return [lon, latitude];
  }
  const a = 6378245.0;
  const ee = 0.00669342162296594323;
  let dLat = transformLat(lon - 105.0, latitude - 35.0);
  let dLng = transformLng(lon - 105.0, latitude - 35.0);
  const radLat = (latitude / 180.0) * Math.PI;
  let magic = Math.sin(radLat);
  magic = 1 - (ee * magic * magic);
  const sqrtMagic = Math.sqrt(magic);
  dLat = (dLat * 180.0) / (((a * (1 - ee)) / (magic * sqrtMagic)) * Math.PI);
  dLng = (dLng * 180.0) / ((a / sqrtMagic) * Math.cos(radLat) * Math.PI);
  return [lon + dLng, latitude + dLat];
}

export function gcj02ToWgs84(lng, lat) {
  const lon = Number(lng);
  const latitude = Number(lat);
  if (!Number.isFinite(lon) || !Number.isFinite(latitude) || outOfChina(lon, latitude)) {
    return [lon, latitude];
  }
  const [mgLng, mgLat] = wgs84ToGcj02(lon, latitude);
  return [lon * 2 - mgLng, latitude * 2 - mgLat];
}

// Backend remains WGS84; convert only at map display boundary.
export function toDisplayLatLng(wgs84LngLat, mapProvider = "") {
  if (!Array.isArray(wgs84LngLat) || wgs84LngLat.length < 2) return null;
  let lng = Number(wgs84LngLat[0]);
  let lat = Number(wgs84LngLat[1]);
  if (isTencentProvider(mapProvider)) {
    [lng, lat] = wgs84ToGcj02(lng, lat);
  }
  return [lat, lng];
}

// Convert map click/input coordinates back to backend WGS84.
export function fromDisplayLatLng(displayLatLng, mapProvider = "") {
  if (!Array.isArray(displayLatLng) || displayLatLng.length < 2) return null;
  let lng = Number(displayLatLng[1]);
  let lat = Number(displayLatLng[0]);
  if (isTencentProvider(mapProvider)) {
    [lng, lat] = gcj02ToWgs84(lng, lat);
  }
  return [lng, lat];
}
