export default function Header() {
  return (
    <header className="fixed top-0 left-[240px] right-0 h-14 bg-[#f7faf6] border-b border-[#c1c8c8] z-40 flex items-center justify-between px-8">
      <div className="flex items-center gap-2">
        <span className="font-sans text-[11px] leading-[14px] text-[#414849]">
          Registry Ledger ID:
        </span>
        <span className="font-mono text-[12px] leading-[16px] text-[#181c1a] font-medium">
          CR-2024-8891-AD
        </span>
      </div>

      <div className="flex items-center gap-6">
        <div className="flex items-center gap-1.5 font-mono text-[12px] leading-[16px] text-[#39684a]">
          <span className="w-2 h-2 rounded-full bg-[#39684a] inline-block animate-pulse"></span>
          Model Registry Synced
        </div>
        <div className="w-7 h-7 rounded-full bg-[#072427] flex items-center justify-center text-white text-[12px] font-semibold">
          A
        </div>
      </div>
    </header>
  );
}
