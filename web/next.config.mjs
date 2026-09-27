/**
 * Static export. The demo has no server: every number was computed by the
 * Python pipeline, stored in Postgres, and frozen into data/demo.json by
 * scripts/export_demo.py. A page that could compute could compute something
 * different from what the repo says it computed.
 */
const nextConfig = {
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
