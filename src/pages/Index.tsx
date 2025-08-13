import { ChatInterface } from "@/components/ChatInterface";
import heroImage from "@/assets/banking-hero.jpg";
import { useTheme } from "next-themes";

const Index = () => {
  const { theme } = useTheme();

  return (
    <div
      className="min-h-screen bg-background relative overflow-hidden transition-colors duration-300"
      style={{
        backgroundImage: theme === 'dark'
          ? `linear-gradient(rgba(15, 23, 42, 0.9), rgba(15, 23, 42, 0.9)), url(${heroImage})`
          : `linear-gradient(rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.95)), url(${heroImage})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundAttachment: 'fixed'
      }}
    >
      <div className="absolute inset-0 bg-gradient-surface opacity-50"></div>
      <div className="relative z-10">
        <ChatInterface />
      </div>
    </div>
  );
};

export default Index;
