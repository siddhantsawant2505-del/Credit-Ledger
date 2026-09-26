import type { Metadata } from "next";
import Sidebar from "@/components/Sidebar";
import Header from "@/components/Header";
import "./globals.css";

export const metadata: Metadata = {
  title: "Credit Ledger — Risk Model Evaluation Suite",
  description: "Institutional credit risk evaluation, model comparison, and default prediction engine.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full bg-[#f7faf6] text-[#181c1a] antialiased">
        <Sidebar />
        <Header />
        <main className="pl-[240px] pt-14 min-h-screen bg-[#f7faf6]">
          <div className="max-w-[1100px] p-12">{children}</div>
        </main>
      </body>
    </html>
  );
}
