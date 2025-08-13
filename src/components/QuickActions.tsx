import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { CreditCard, PiggyBank, TrendingUp, HelpCircle, ArrowRightLeft, Shield, ChevronDown, ChevronUp, ArrowLeft } from "lucide-react";
import { useState } from "react";

interface QuickActionsProps {
  onActionClick: (action: string) => void;
}

interface CategoryQuestions {
  [key: string]: string[];
}

const quickActions = [
  {
    icon: CreditCard,
    label: "NUST Accounts",
    category: "accounts",
    action: "What are the features of NUST Maximiser Savings Account?"
  },
  {
    icon: ArrowRightLeft,
    label: "Transfers",
    category: "transfers",
    action: "What are the processing charges for PMYB & ALS?"
  },
  {
    icon: PiggyBank,
    label: "Loans",
    category: "loans",
    action: "Can applicant avail clean loan in NUST Sahar Finance?"
  },
  {
    icon: TrendingUp,
    label: "Digital Banking",
    category: "digital",
    action: "Are there any Credit and Debit limits in NUST Freelancer Digital Account?"
  },
  {
    icon: Shield,
    label: "Security",
    category: "security",
    action: "What are the free facilities associated with Roshan Digital Account?"
  },
  {
    icon: HelpCircle,
    label: "Support",
    category: "support",
    action: "What are the Loan Limits of NUST Rice Finance?"
  }
];

const categoryQuestions: CategoryQuestions = {
  accounts: [
    "What are the features of NUST Maximiser Savings Account?",
    "What are the minimum balance requirements for NUST accounts?",
    "How can I open a NUST Freelancer Digital Account?",
    "What are the benefits of NUST Sahar Finance account?",
    "What documents are required to open a NUST account?"
  ],
  transfers: [
    "What are the processing charges for PMYB & ALS?",
    "What are the transfer limits for digital banking?",
    "How long do international transfers take?",
    "What are the charges for domestic wire transfers?",
    "How can I set up recurring transfers?"
  ],
  loans: [
    "Can applicant avail clean loan in NUST Sahar Finance?",
    "What are the Loan Limits of NUST Rice Finance?",
    "What are the eligibility criteria for personal loans?",
    "What documents are required for loan application?",
    "What are the current interest rates for different loan types?"
  ],
  digital: [
    "Are there any Credit and Debit limits in NUST Freelancer Digital Account?",
    "How to activate mobile banking services?",
    "What are the features of the NUST mobile app?",
    "How to reset my digital banking password?",
    "What are the security features of digital banking?"
  ],
  security: [
    "What are the free facilities associated with Roshan Digital Account?",
    "How to report a lost or stolen card?",
    "What are the fraud protection measures?",
    "How to enable two-factor authentication?",
    "What should I do if I suspect unauthorized access?"
  ],
  support: [
    "What are the customer service hours?",
    "How to file a complaint or grievance?",
    "What are the contact details for different services?",
    "How to request account statements?",
    "How to update my contact information?"
  ]
};

export const QuickActions = ({ onActionClick }: QuickActionsProps) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  const handleCategoryClick = (category: string) => {
    setSelectedCategory(category);
  };

  const handleBackToCategories = () => {
    setSelectedCategory(null);
  };

  const handleQuestionClick = (question: string) => {
    onActionClick(question);
    setSelectedCategory(null); // Retour aux catégories après sélection
  };

  return (
    <Card className="bg-card/50 border-border/50 backdrop-blur-sm">
      <div className="p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            {selectedCategory && (
              <Button
                variant="ghost"
                size="sm"
                onClick={handleBackToCategories}
                className="h-6 w-6 p-0 hover:bg-primary/10 text-muted-foreground hover:text-foreground transition-colors"
              >
                <ArrowLeft className="h-4 w-4" />
              </Button>
            )}
            <h3 className="text-sm font-medium text-foreground">
              {selectedCategory ?
                `Questions suggérées - ${quickActions.find(a => a.category === selectedCategory)?.label}` :
                "Actions rapides"
              }
            </h3>
          </div>
          {!selectedCategory && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsExpanded(!isExpanded)}
              className="h-6 w-6 p-0 hover:bg-primary/10 text-muted-foreground hover:text-foreground transition-colors"
            >
              {isExpanded ? (
                <ChevronUp className="h-4 w-4" />
              ) : (
                <ChevronDown className="h-4 w-4" />
              )}
            </Button>
          )}
        </div>

        {selectedCategory ? (
          // Affichage des questions suggérées pour la catégorie sélectionnée
          <div className="space-y-2">
            {categoryQuestions[selectedCategory]?.map((question, index) => (
              <Button
                key={index}
                variant="ghost"
                onClick={() => handleQuestionClick(question)}
                className="w-full text-left justify-start p-3 h-auto hover:bg-primary/5 border border-border/30 hover:border-primary/30 transition-all duration-200 text-muted-foreground hover:text-foreground"
              >
                <span className="text-sm leading-relaxed">{question}</span>
              </Button>
            ))}
          </div>
        ) : (
          // Affichage des catégories (comportement original)
          <>
            {isExpanded && (
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2 transition-all duration-300">
                {quickActions.map((action, index) => {
                  const IconComponent = action.icon;
                  return (
                    <Button
                      key={index}
                      variant="ghost"
                      onClick={() => handleCategoryClick(action.category)}
                      className="h-auto p-3 flex flex-col gap-2 hover:bg-primary/5 border border-border/30 hover:border-primary/30 transition-all duration-200 group text-muted-foreground hover:text-foreground"
                    >
                      <IconComponent className="h-5 w-5 text-primary group-hover:scale-110 transition-transform" />
                      <span className="text-xs text-center leading-tight">{action.label}</span>
                    </Button>
                  );
                })}
              </div>
            )}

            {!isExpanded && (
              <div className="flex gap-2 overflow-x-auto pb-2">
                {quickActions.slice(0, 4).map((action, index) => {
                  const IconComponent = action.icon;
                  return (
                    <Button
                      key={index}
                      variant="ghost"
                      onClick={() => handleCategoryClick(action.category)}
                      className="flex-shrink-0 h-auto p-2 flex items-center gap-2 hover:bg-primary/5 border border-border/30 hover:border-primary/30 transition-all duration-200 group text-muted-foreground hover:text-foreground"
                    >
                      <IconComponent className="h-4 w-4 text-primary group-hover:scale-110 transition-transform" />
                      <span className="text-xs whitespace-nowrap">{action.label}</span>
                    </Button>
                  );
                })}
                <Button
                  variant="ghost"
                  onClick={() => setIsExpanded(true)}
                  className="flex-shrink-0 h-auto p-2 flex items-center gap-2 hover:bg-primary/5 border border-border/30 hover:border-primary/30 transition-all duration-200 text-muted-foreground hover:text-foreground"
                >
                  <span className="text-xs">+{quickActions.length - 4} autres</span>
                </Button>
              </div>
            )}
          </>
        )}
      </div>
    </Card>
  );
};