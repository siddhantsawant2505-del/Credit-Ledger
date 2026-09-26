"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Sidebar() {
  const pathname = usePathname();

  const navItems = [
    { label: "Overview", href: "/", pathKey: "overview" },
    { label: "Predict & results", href: "/predict-results", pathKey: "predict-results" },
    { label: "Model comparison", href: "/model-comparison", pathKey: "model-comparison" },
    { label: "Explainability", href: "/explainability", pathKey: "explainability" },
  ];

  return (
    <aside className="fixed left-0 top-0 h-full w-[240px] bg-[#f7faf6] border-r border-[#c1c8c8] z-50 flex flex-col justify-between select-none">
      <div className="pt-6">
        {/* Brand Dossier Header */}
        <div className="px-6 pb-6 border-b border-[#c1c8c8]">
          <div className="font-serif text-[18px] leading-[24px] font-medium text-[#072427] tracking-tight">
            Credit Ledger
          </div>
          <div className="font-sans text-[12px] leading-[16px] text-[#414849] mt-1">
            Risk Model Evaluation Suite
          </div>
        </div>

        {/* Navigation Section */}
        <nav className="mt-4 flex flex-col">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`px-6 py-2 text-[13px] font-medium transition-colors block border-l-2 ${
                  isActive
                    ? "text-[#1f3a3d] border-[#1f3a3d] bg-[#f1f4f1] font-medium"
                    : "text-[#414849] border-transparent hover:text-[#181c1a] hover:bg-[#ecefeb]"
                }`}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Footer Metadata */}
      <div className="p-6 border-t border-[#c1c8c8] bg-[#f7faf6]">
        <div className="font-mono text-[12px] leading-[16px] text-[#181c1a] text-opacity-90">
          Validation Suite v2.4
        </div>
        <div className="font-sans text-[12px] leading-[16px] text-[#414849] mt-1">
          Quarterly Cohort 2024-Q3
        </div>
        <div className="mt-4 pt-4 border-t border-[#c1c8c8] flex items-center justify-between">
          <span className="font-sans text-[11px] leading-[14px] text-[#414849] font-medium">
            Auditor Mode
          </span>
          <div className="w-7 h-7 rounded-full bg-[#072427] flex items-center justify-center text-white text-[12px] font-semibold">
            A
          </div>
        </div>
      </div>
    </aside>
  );
}
