import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = { title: "Hydro Intelligence Agent", description: "Hydropower generation forecasting dashboard" };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
