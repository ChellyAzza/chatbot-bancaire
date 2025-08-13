import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { CreditCard, PiggyBank, TrendingUp, HelpCircle, ArrowRightLeft, Shield, Menu, X } from "lucide-react";
import { useState } from "react";

interface QuickActionsSidebarProps {
  onActionClick: (action: string) => void;
  isOpen?: boolean;
  onClose?: () => void;
}

const quickActions = [
  {
    icon: CreditCard,
    label: "NUST Accounts",
    action: "What are the features of NUST Maximiser Savings Account?"
  },
  {
    icon: ArrowRightLeft,
    label: "Transfers",
    action: "What are the processing charges for PMYB & ALS?"
  },
  {
    icon: PiggyBank,
    label: "Loans",
    action: "Can applicant avail clean loan in NUST Sahar Finance?"
  },
  {
    icon: TrendingUp,
    label: "Digital Banking",
    action: "Are there any Credit and Debit limits in NUST Freelancer Digital Account?"
  },
  {
    icon: Shield,
    label: "Security",
    action: "What are the free facilities associated with Roshan Digital Account?"
  },
  {
    icon: HelpCircle,
    label: "Support",
    action: "What are the Loan Limits of NUST Rice Finance?"
  }
];

export const QuickActionsSidebar = ({ onActionClick, isOpen = true, onClose }: QuickActionsSidebarProps) => {

  return (
    <>
      {/* Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50"
          onClick={onClose}
        />
      )}

      {/* Sidebar */}
      <div className={`fixed top-0 right-0 h-full w-80 bg-background border-l border-border/50 z-50 transform transition-transform duration-300 ${
        isOpen ? 'translate-x-0' : 'translate-x-full'
      }`}>
        <div className="p-4 h-full flex flex-col">
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-foreground">Actions rapides</h3>
            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              className="hover:bg-primary/10 text-muted-foreground hover:text-foreground transition-colors"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>

          {/* Actions */}
          <div className="flex-1 space-y-3">
            {quickActions.map((action, index) => {
              const IconComponent = action.icon;
              return (
                <Card key={index} className="bg-card/50 border-border/50">
                  <Button
                    variant="ghost"
                    onClick={() => {
                      onActionClick(action.action);
                      onClose?.();
                    }}
                    className="w-full h-auto p-4 flex items-start gap-3 hover:bg-primary/5 transition-all duration-200 group text-muted-foreground hover:text-foreground"
                  >
                    <IconComponent className="h-5 w-5 text-primary group-hover:scale-110 transition-transform flex-shrink-0 mt-0.5" />
                    <div className="text-left">
                      <div className="font-medium text-sm">{action.label}</div>
                      <div className="text-xs text-muted-foreground mt-1 line-clamp-2">
                        {action.action}
                      </div>
                    </div>
                  </Button>
                </Card>
              );
            })}
          </div>

          {/* Footer */}
          <div className="mt-6 pt-4 border-t border-border/50">
            <p className="text-xs text-muted-foreground text-center">
              Cliquez sur une action pour poser la question
            </p>
          </div>
        </div>
      </div>
    </>
  );
};
